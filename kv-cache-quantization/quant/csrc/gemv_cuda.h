#pragma once
#include <torch/extension.h>

torch::Tensor gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size);


torch::Tensor gemv_forward_cuda_outer_dim(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv);


torch::Tensor gemv_forward_cuda_outer_dim_cos(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    torch::Tensor _cosB,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv);


torch::Tensor gemv_forward_cuda_outer_dim_rope(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    torch::Tensor _cosB,
    torch::Tensor _sinB,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv);
