import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse


def stripe_flash_attn_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    assert causal, "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"
    comm = RingComm(process_group)

    out = None
    lse = None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step <= comm.rank:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q[:, 1:, ],
                k[:, :-1, ],
                v[:, :-1, ],
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse,
                                          slice_=(slice(None), slice(1, None)))

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out.to(q.dtype), lse


def stripe_flash_attn_backward(
        process_group,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    assert causal, "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(
        q.shape, dtype=q.dtype, device=q.device
    )
    block_dk_buffer = torch.empty(
        k.shape, dtype=k.dtype, device=k.device
    )
    block_dv_buffer = torch.empty(
        v.shape, dtype=v.dtype, device=v.device
    )
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        shift_causal = step > kv_comm.rank
        if not shift_causal:
            _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                block_dq_buffer,
                block_dk_buffer,
                block_dv_buffer,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )
        else:
            shrink_size = list(q.shape)
            shrink_size[1] -= 1
            block_dq_buffer = torch.empty(shrink_size, dtype=q.dtype, device=q.device)
            block_dk_buffer = torch.empty(shrink_size, dtype=q.dtype, device=q.device)
            block_dv_buffer = torch.empty(shrink_size, dtype=q.dtype, device=q.device)
            _flash_attn_backward(
                dout[:, 1:],
                q[:, 1:],
                k[:, :-1],
                v[:, :-1],
                out[:, 1:],
                softmax_lse[:, :, 1:].contiguous(),
                block_dq_buffer,
                block_dk_buffer,
                block_dv_buffer,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )

        if dq is None:
            dq = block_dq_buffer.to(torch.float32)
            dk = block_dk_buffer.to(torch.float32)
            dv = block_dv_buffer.to(torch.float32)
        else:
            if not shift_causal:
                dq += block_dq_buffer
            else:
                dq[:, 1:] += block_dq_buffer
            d_kv_comm.wait()
            if not shift_causal:
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
            else:
                dk = next_dk
                dv = next_dv
                dk[:, :-1] += block_dk_buffer
                dv[:, :-1] += block_dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class StripeFlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            qkv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_softmax,
            group,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = qkv[:, :, 0].contiguous()
        k = qkv[:, :, 1].contiguous()
        v = qkv[:, :, 2].contiguous()
        out, softmax_lse = stripe_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = stripe_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        dqkv = torch.stack([dq, dk, dv], dim=2)
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None


def stripe_flash_attn_qkvpacked_func(
        qkv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        group=None,
):
    return StripeFlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )