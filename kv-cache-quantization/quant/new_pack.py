import triton
import triton.language as tl
import random
import numpy as np
import torch


def quant_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
    assert len(k.shape) == 4
    shape = k.shape
    B, nh, T, D = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B, nh, num_groups, group_size, D)
    # Quantize
    max_int = 2**bits - 1
    data = k.view(new_shape)
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)
    code = pack_tensor(data, bits, pack_dim=2)
    return code, scale, mn


def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
    shape = v.shape
    assert len(shape) == 4
    assert v.shape[-1] % group_size == 0
    num_groups = shape[-1] // group_size
    new_shape = shape[:-1] + (num_groups, group_size)
    # Quantize
    max_int = 2**bits - 1
    data = v.view(new_shape)
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)
    # Pack
    code = pack_tensor(data, bits, pack_dim=3)
    return code, scale, mn


def unpack_and_dequant_kcache(
    k_code: torch.FloatTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
):
    pack_dim = 2
    assert bits in [2, 4, 8]
    assert len(k_code.shape) == 4
    data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
    shape = data.shape
    num_groups = shape[pack_dim] // group_size
    data = data.view(
        shape[:pack_dim]
        + (
            num_groups,
            group_size,
        )
        + shape[pack_dim + 1 :]
    )
    data = data.to(torch.float16)
    data = data * scale + mn
    return data.view(shape)


def unpack_and_dequant_vcache(
    v_code: torch.FloatTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
):
    assert bits in [2, 4, 8]
    assert len(v_code.shape) == 4
    data = unpack_tensor(v_code, bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(
        shape[:-1]
        + (
            num_groups,
            group_size,
        )
    )
    data = data.to(torch.float16)
    data = data * scale + mn
    return data.view(shape)


def pack_tensor(data, bits, pack_dim):
    # Pack
    shape = data.shape
    feat_per_int = 32 // bits
    assert bits in [2, 4, 8], "Only 2, 4, 8 bits are supported"
    assert (
        shape[pack_dim] % feat_per_int == 0
    ), "Dimension length must be divisible by number of features per int"
    # BS, nh, T, nd // 16 # 16 is for 2bit
    code = torch.zeros(
        shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1 :],
        dtype=torch.int32,
        device=data.device,
    )
    i = 0
    row = 0
    unpacked_indices = [slice(None)] * len(data.shape)
    packed_indices = [slice(None)] * len(data.shape)
    while row < code.shape[pack_dim]:
        packed_indices[pack_dim] = row
        for j in range(i, i + (32 // bits)):
            unpacked_indices[pack_dim] = j
            code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
        i += 32 // bits
        row += 1
    return code


def unpack_tensor(v_code: torch.FloatTensor, bits: int, pack_dim: int):
    assert bits in [2, 4, 8]
    shape = v_code.shape
    feat_per_int = 32 // bits
    new_shape = (
        shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1 :]
    )
    unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
    i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
    j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
    num = 0xFF >> (8 - bits)
    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i
    if pack_dim == 2:
        unpacked_v_code = (
            (v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)
        ) & num
    elif pack_dim == 3:
        unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
    else:
        raise NotImplementedError
    return unpacked_v_code


@triton.jit
def _pack_along_last_dim(
    bits: tl.constexpr,
    intensor_ptr,
    code_ptr,
    N,
    num_feats: tl.constexpr,
    feat_per_int: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    num_int_per_y_dim = num_feats // feat_per_int
    bid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_start = (
        intensor_ptr + offs_N * num_feats + yid * feat_per_int
    )  # offset of the first element at current tile
    packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    for i in range(feat_per_int):
        ptr = block_start + i
        element = tl.load(ptr, mask=offs_N < N, other=0.0)
        element = element << (i * bits)
        # Combine the value using bitwise OR
        packed = packed | element
    tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)


@triton.jit
def _minmax_along_last_dim(
    x_ptr,
    mn_ptr,
    mx_ptr,
    total_elements: tl.constexpr,
    N: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
    mask = offsets < total_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mx_val = tl.max(x, axis=1)
    mn_val = tl.min(x, axis=1)
    # tl.device_print('shape', mn_val[:, None].shape)
    tl.store(mn_ptr + offsets_b, mn_val, mask=offsets_b < N * num_groups)
    tl.store(mx_ptr + offsets_b, mx_val, mask=offsets_b < N * num_groups)


def triton_quantize_and_pack_along_last_dim(
    data: torch.Tensor, group_size: int, bit: int,
):
    assert len(data.shape) == 4
    shape = data.shape
    B, nh, D, T = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B * nh * D, num_groups, group_size)
    scale_mn_shape = B, nh, D, num_groups
    # Quantize
    data = data.reshape(new_shape)
    mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(data.shape[0] * data.shape[1], BLOCK_SIZE_N),)
    with torch.cuda.device(data.device):
        _minmax_along_last_dim[grid](
            data,
            mn,
            mx,
            data.numel(),
            data.shape[0],
            num_groups,
            group_size,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=8,
        )
    # mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
    # mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
    scale = (mx - mn) / (2**bit - 1)
    data = data - mn.unsqueeze(-1)
    data.div_(scale.unsqueeze(-1))
    data = data.clamp_(0, 2**bit - 1).round_().to(torch.int32)  # !!! nan will be converted to 0, which is good
    data = data.view(-1, T)
    feat_per_int = 32 // bit
    packshape = (
        np.prod(shape[:-1]),
        shape[-1] // feat_per_int,
    )
    code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
    grid = lambda meta: (
        triton.cdiv(data.shape[0], BLOCK_SIZE_N),
        data.shape[1] // feat_per_int,
    )
    with torch.cuda.device(data.device):
        _pack_along_last_dim[grid](
            bit,
            data,
            code,
            data.shape[0],
            data.shape[1],
            feat_per_int,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=8,
        )
    return (
        code.view(B, nh, D, -1),
        scale.reshape(scale_mn_shape),
        mn.reshape(scale_mn_shape),
    )


def triton_quantize_and_pack_along_last_dim_with_dequant(data: torch.Tensor, group_size: int, bit: int):
    assert len(data.shape) == 4
    shape = data.shape
    B, nh, D, T = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B * nh * D, num_groups, group_size)
    scale_mn_shape = B, nh, D, num_groups
    # Quantize
    data = data.reshape(new_shape)
    mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
    _minmax_along_last_dim[grid](data, mn, mx,
                             data.numel(), data.shape[0], num_groups, group_size,
                             BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
    # mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
    # mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
    scale = (mx - mn) / (2 ** bit - 1)
    quant_data = data - mn.unsqueeze(-1)
    quant_data.div_(scale.unsqueeze(-1))
    quant_data = quant_data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
    data_dequant = quant_data.to(scale.dtype) * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    quant_data = quant_data.view(-1, T)
    feat_per_int = 32 // bit
    packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
    code = torch.zeros(*packshape, device=quant_data.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(quant_data.shape[0], BLOCK_SIZE_N), quant_data.shape[1] // feat_per_int,)  # !!! use quant_data.shape[1]
    _pack_along_last_dim[grid](bit, quant_data, code, quant_data.shape[0], 
                                quant_data.shape[1], feat_per_int, 
                                BLOCK_SIZE_N=BLOCK_SIZE_N, 
                                num_warps=8)
    return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape), data_dequant.reshape(shape)


def unpack_and_dequant_along_last_dim(
    v_code: torch.FloatTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
):
    assert bits in [2, 4, 8]
    assert len(v_code.shape) == 4
    data = unpack_tensor(v_code, bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(
        shape[:-1]
        + (
            num_groups,
            group_size,
        )
    )
    data = data.to(torch.float16)
    data = data * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return data.view(shape)


def generate_At_inv(quant_group_size, my_Qhat, lamb=1, tol=1e-7):
    """
    Generate a list of T matrices where the t-th matrix has dimension (t*g, t*g).

    Parameters:
    - quant_group_size (int): Factor for matrix dimension scaling
    - lamb (float): Scaling factor for the final term
    - my_Qhat (torch.Tensor): A matrix of size (d, d)

    Returns:
    - List[torch.Tensor]: List of int(head_dim/quant_group_size) matrices
    """

    bs, kv_nh, subspace_dim, head_dim = my_Qhat.shape
    # T = int(head_dim / quant_group_size)
    T = (head_dim+quant_group_size-1)//quant_group_size
    matrices = [None] * T
    device = my_Qhat.device
    I = torch.eye(head_dim, device=device)
    # Initialize A_T
    A_T = I.expand(bs, kv_nh, head_dim, head_dim) + lamb * torch.matmul(
        my_Qhat.transpose(-1, -2), my_Qhat
    )
    matrices[T - 1] = A_T

    # Recursive computation of A_{t} from A_{t+1}
    for t in range(T - 1, 0, -1):
        current_dim = t * quant_group_size

        # Extract M_{t+1}, N_{t+1}, and O_{t+1}
        M_t1 = A_T[:, :, :current_dim, :current_dim]  # Top-left square matrix
        N_t1 = A_T[:, :, current_dim : current_dim + quant_group_size, :current_dim]  # Bottom-left matrix
        O_t1 = A_T[:, :, current_dim : current_dim + quant_group_size, current_dim : current_dim + quant_group_size]  # Bottom-right square matrix

        # Compute A_t
        I_mat = torch.eye(quant_group_size, device=device)
        O_t1_inv = torch.inverse(O_t1 + tol * I_mat.expand(bs, kv_nh, quant_group_size, quant_group_size))
        A_t = M_t1 - torch.matmul(N_t1.transpose(-1, -2), torch.matmul(O_t1_inv, N_t1))
        # matrices[t - 1] = A_t
        matrices[t - 1] = A_t[:, :, :, -quant_group_size:]

        # Update A_T for the next iteration
        A_T = A_t

    return matrices


def squat_lagrangian(
    key_states_full_trans, quant_group_size, seq_group_size, k_bits, Ainv_t, P_inv
):
    # key_states_full_trans: [b, nh/4, d, t]
    # group_size: 32
    # k_bits: 2, 4
    key_states_quant_trans, key_scale_trans, key_mn_trans = new_quant(
        key_states_full_trans.transpose(2, 3),
        quant_group_size,
        seq_group_size,
        k_bits,
        Ainv_t,
        P_inv,
    )
    return key_states_quant_trans, key_scale_trans, key_mn_trans


def new_quant(key_states, quant_group_size, seq_group_size, k_bits, Ainv_t, P_inv):
    # TODO: give ht instead of Ainv_t as input

    dtype = key_states.dtype
    # P_inv = torch.inverse(Ainv_t[-1])

    bsz, nh, seq_len, hidden_dim = key_states.shape

    # hidden_dim needs to be divisible by quant_group_size
    T = (hidden_dim+quant_group_size-1)//quant_group_size
    # T = hidden_dim // quant_group_size

    key_states_quant_trans, key_scale_trans, key_mn_trans = [], [], []
    
    group = key_states  # Extract the group

    for i in range(T):

        # key_states_quant_trans_this_quant_group, key_scale_trans_this_quant_group, key_mn_trans_this_quant_group = triton_quantize_and_pack_along_last_dim(group[:, :, :, i * quant_group_size : (i + 1) * quant_group_size].transpose(2, 3), seq_group_size, k_bits)
        # dequantized_trans = unpack_and_dequant_along_last_dim(key_states_quant_trans_this_quant_group, key_scale_trans_this_quant_group, key_mn_trans_this_quant_group, seq_group_size, k_bits)
        key_states_quant_trans_this_quant_group, key_scale_trans_this_quant_group, key_mn_trans_this_quant_group, dequantized_trans = triton_quantize_and_pack_along_last_dim_with_dequant(group[:, :, :, i * quant_group_size : (i + 1) * quant_group_size].transpose(2, 3), seq_group_size, k_bits)
        dequantized = dequantized_trans.transpose(2, 3)

        if i < T - 1:
            d_vec = (
                dequantized
                - group[:, :, :, i * quant_group_size : (i + 1) * quant_group_size]
            ).float()
            H_t = Ainv_t[i]

            # H_t = H_t[:, :, :, -quant_group_size:]
            B_t = P_inv[
                :, :, (i + 1) * quant_group_size :, : (i + 1) * quant_group_size
            ]

            update = torch.matmul(
                torch.matmul(d_vec, H_t.transpose(-2, -1)), B_t.transpose(-2, -1)
            )

            group[:, :, :, (i + 1) * quant_group_size :] = (
                group[:, :, :, (i + 1) * quant_group_size :] + update
            )
        
        key_states_quant_trans.append(key_states_quant_trans_this_quant_group)
        key_scale_trans.append(key_scale_trans_this_quant_group)
        key_mn_trans.append(key_mn_trans_this_quant_group)

    key_states_quant_trans = torch.cat(key_states_quant_trans, dim=2)
    key_scale_trans = torch.cat(key_scale_trans, dim=2)
    key_mn_trans = torch.cat(key_mn_trans, dim=2)

    return key_states_quant_trans, key_scale_trans, key_mn_trans


def block_power_iteration(A, subspace_dim, num_iterations=10, tol=1e-6):
    """
    Approximates the top `subspace_dim` singular vectors of A using Block Power Iteration.

    Parameters:
    A : torch.Tensor
        Input matrix of shape (num_head, seq_len, head_dim).
    subspace_dim : int
        Number of dominant singular vectors to compute.
    num_iterations : int, optional
        Maximum number of iterations (default: 100).
    tol : float, optional
        Convergence tolerance (default: 1e-6).

    Returns:
    Vh_subspace : torch.Tensor
        Approximate top `subspace_dim` right singular vectors (num_head, subspace_dim, head_dim).
    S_subspace : torch.Tensor
        Corresponding singular values (num_head, subspace_dim).
    """
    num_head, seq_len, head_dim = A.shape

    # Initialize a random matrix
    Q = torch.randn(num_head, head_dim, subspace_dim, device=A.device, dtype=A.dtype)
    Q, _ = torch.linalg.qr(Q)  # Orthonormalize

    for _ in range(num_iterations):
        Z = torch.matmul(A.transpose(1, 2), torch.matmul(A, Q))  # Equivalent to A^T A Q
        Q_new, _ = torch.linalg.qr(Z)  # Re-orthonormalize
        
        # # Check convergence (optional)
        # if torch.norm(Q - Q_new) < tol:
        #     break

        Q = Q_new

    # Compute singular values
    S_subspace = torch.norm(torch.matmul(A, Q), dim=1)  # Singular values
    
    return S_subspace, Q.transpose(1, 2)  # Return Vh_subspace
