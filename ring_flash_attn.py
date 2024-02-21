import torch
import torch.distributed as dist
from raw_flash_attn import raw_flash_attn_forward, raw_flash_attn_backward


def ring_flash_attn_forward(
    local_q,
    local_k,
    local_v,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_q = local_q.contiguous()
    local_k = local_k.contiguous()
    local_v = local_v.contiguous()

    ks = [torch.zeros_like(local_k) for _ in range(world_size)]
    vs = [torch.zeros_like(local_v) for _ in range(world_size)]

    dist.all_gather(ks, local_k)
    dist.all_gather(vs, local_v)

    out = None
    lse = None
    end = rank + 1 if causal else world_size
    for i in range(end):
        local_causal = causal and i == rank
        k = ks[i]
        v = vs[i]
        block_out, block_lse, _ = raw_flash_attn_forward(
            local_q,
            k,
            v,
            dropout_p,
            causal=local_causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_softmax=True,
        )
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)
        if out is None:
            out = block_out
            lse = block_lse
        else:
            new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
            out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
            lse = new_lse

    out = out.to(torch.bfloat16)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_backward(
    local_dout,
    local_q,
    local_k,
    local_v,
    local_out,
    softmax_lse,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_dout = local_dout.contiguous()
    local_q = local_q.contiguous()
    local_k = local_k.contiguous()
    local_v = local_v.contiguous()
    local_out = local_out.contiguous()

    ks = [torch.zeros_like(local_k) for _ in range(world_size)]
    vs = [torch.zeros_like(local_v) for _ in range(world_size)]

    dist.all_gather(ks, local_k)
    dist.all_gather(vs, local_v)

    local_dq = None
    dks = []
    dvs = []
    end = rank + 1 if causal else world_size
    for i in range(end):
        local_causal = causal and i == rank
        k = ks[i]
        v = vs[i]
        block_dq, block_dk, block_dv = raw_flash_attn_backward(
            local_dout,
            local_q,
            k,
            v,
            local_out,
            softmax_lse,
            rng_state=None,
            dropout_p=dropout_p,
            causal=local_causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            softmax_scale=None,
        )
        block_dq = block_dq.to(torch.float32)
        block_dk = block_dk.to(torch.float32)
        block_dv = block_dv.to(torch.float32)

        if local_dq is None:
            local_dq = block_dq
        else:
            local_dq += block_dq
        dks.append(block_dk)
        dvs.append(block_dv)

    for i in range(end, world_size):
        dks.append(torch.zeros_like(block_dk))
        dvs.append(torch.zeros_like(block_dv))

    dks = torch.cat(dks, dim=1)
    dvs = torch.cat(dvs, dim=1)
    dist.all_reduce(dks)
    dist.all_reduce(dvs)

    local_dk = dks.chunk(world_size, dim=1)[rank]
    local_dv = dvs.chunk(world_size, dim=1)[rank]

    return local_dq, local_dk, local_dv


class RingFlashAttnQKVPackedFunc(torch.autograd.Function):
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
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = qkv[:, :, 0].contiguous()
        k = qkv[:, :, 1].contiguous()
        v = qkv[:, :, 2].contiguous()
        out, softmax_lse = ring_flash_attn_forward(
            q,
            k,
            v,
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
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        dqkv = torch.stack([dq, dk, dv], dim=2)
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None


def ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return RingFlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )