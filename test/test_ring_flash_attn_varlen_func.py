from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
import torch
import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_varlen_qkvpacked_func, ring_flash_attn_varlen_func
from utils import log, set_seed


# def extract_local(value, cu_seqlens, rank, world_size):
#     local_values = []
#     for i in range(len(cu_seqlens) - 1):
#         start, end = cu_seqlens[i], cu_seqlens[i + 1]
#         local_value = value[start:end].chunk(world_size, dim=0)[rank].detach().clone()
#         local_values.append(local_value)
#     return torch.cat(local_values, dim=0).contiguous()

def extract_local(value, cu_seqlens, rank, world_size):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        local_value = value[start:end].chunk(2 * world_size, dim=0)
        local_values.extend(
            [
                local_value[rank].detach().clone(),
                local_value[2 * world_size - 1 - rank].detach().clone(),
            ]
        )
    return torch.cat(local_values, dim=0).contiguous()


def extract_lse(lse, cu_seqlens):
    values = []
    if lse.dim() == 2:
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            value = lse[:, start:end]
            values.append(value)
    else:
        assert lse.dim() == 3
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            value = lse[i, :, : end - start]
            values.append(value)
    return values


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    nheads = 5
    d = 128
    dropout_p = 0
    causal = False
    deterministic = False

    cu_seqlens = [0, 120, 1248, 4232]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
    total_length = cu_seqlens[-1]
    num_seq = len(cu_seqlens) - 1
    
    cu_seqlens_k = [x * 4 for x in cu_seqlens]
    # cu_seqlens_k = cu_seqlens
    cu_seqlens_tensor_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    max_seqlen_k = (cu_seqlens_tensor_k[1:] - cu_seqlens_tensor_k[:-1]).max().item()
    total_length_k = cu_seqlens_k[-1]

    assert torch.all(cu_seqlens_tensor % world_size == 0) # the seqlens should be divisible by world_size
    assert d % 8 == 0 # d should be divisible by 8

    # qkv = torch.randn(
    #     total_length, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    # )
    # dist.broadcast(qkv, src=0)
    
    q = torch.randn(total_length, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_length_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_length_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    

    dout = torch.randn(total_length, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_cu_seqlens_tensor = cu_seqlens_tensor // world_size
    local_max_seqlen = max_seqlen // world_size
    local_cu_seqlens_tensor_k = cu_seqlens_tensor_k // world_size
    local_max_seqlen_k = max_seqlen_k // world_size

    # local_qkv = extract_local(qkv, cu_seqlens, rank, world_size)
    # local_qkv.requires_grad = True
    local_q = extract_local(q, cu_seqlens, rank, world_size)
    local_k = extract_local(k, cu_seqlens_k, rank, world_size)
    local_v = extract_local(v, cu_seqlens_k, rank, world_size)
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    local_dout = extract_local(dout, cu_seqlens, rank, world_size)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    # out, lse, _ = flash_attn_varlen_qkvpacked_func(
    #     qkv,
    #     cu_seqlens_tensor,
    #     max_seqlen,
    #     dropout_p=dropout_p,
    #     causal=causal,
    #     window_size=(-1, -1),
    #     alibi_slopes=None,
    #     deterministic=deterministic,
    #     return_attn_probs=True,
    # )
    out, lse, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_tensor,
        cu_seqlens_tensor_k,
        max_seqlen,
        max_seqlen_k,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = extract_local(out, cu_seqlens, rank, world_size)
    lse_list = extract_lse(lse, cu_seqlens)

    # ring_out, ring_lse, _ = ring_flash_attn_varlen_qkvpacked_func(
    #     local_qkv,
    #     local_cu_seqlens_tensor,
    #     local_cu_seqlens_tensor,
    #     local_max_seqlen,
    #     local_max_seqlen,
    #     dropout_p=dropout_p,
    #     causal=causal,
    #     window_size=(-1, -1),
    #     alibi_slopes=None,
    #     deterministic=deterministic,
    #     return_attn_probs=True,
    # )
    ring_out, ring_lse, _ = ring_flash_attn_varlen_func(
        local_q,
        local_k,
        local_v,
        local_cu_seqlens_tensor,
        local_cu_seqlens_tensor_k,
        local_max_seqlen,
        local_max_seqlen_k,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    ring_lse_list = extract_lse(ring_lse, local_cu_seqlens_tensor.tolist())

    log("out", out, rank0_only=True)
    log("out diff", local_out - ring_out)

    for lse, ring_lse in zip(lse_list, ring_lse_list):
        local_lse = lse.chunk(world_size, dim=-1)[rank]
        log("lse", lse, rank0_only=True)
        log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    # dqkv = qkv.grad
    # local_dqkv = extract_local(dqkv, cu_seqlens, rank, world_size)
    dq, dk, dv = q.grad, k.grad, v.grad
    local_dq = extract_local(dq, cu_seqlens, rank, world_size)
    local_dk = extract_local(dk, cu_seqlens_k, rank, world_size)
    local_dv = extract_local(dv, cu_seqlens_k, rank, world_size)

    ring_out.backward(local_dout)
    # ring_dqkv = local_qkv.grad
    ring_dq, ring_dk, ring_dv = local_q.grad, local_k.grad, local_v.grad
    # ring_dqkv = torch.stack([ring_dq, ring_dk, ring_dv], dim=1)

    # log("local_dqkv", local_dqkv)
    # log("dq diff", local_dqkv[:, 0] - ring_dqkv[:, 0])
    # log("dk diff", local_dqkv[:, 1] - ring_dqkv[:, 1])
    # log("dv diff", local_dqkv[:, 2] - ring_dqkv[:, 2])
    log("local_dq", local_dq)
    log("local_dk", local_dk)
    log("local_dv", local_dv)
    log("ring_dq", ring_dq)
    log("ring_dk", ring_dk)
    log("ring_dv", ring_dv)
    log("local_dq - ring_dq", local_dq - ring_dq)
    log("local_dk - ring_dk", local_dk - ring_dk)
    log("local_dv - ring_dv", local_dv - ring_dv)
    dist.barrier()
