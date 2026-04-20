import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_along_last_dim, squat_lagrangian, generate_At_inv, block_power_iteration
from quant.matmul import cuda_bmm_fA_qB_outer, cuda_bmm_fA_qB_outer_cos_sin, cuda_bmm_fA_qB_outer_rope

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from transformers.utils import is_flash_attn_greater_or_equal_2_10 #, is_flash_attn_2_available
from typing import Dict, Any, Union, Callable

from transformers.generation.utils import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationMode,
    GenerateOutput,
)
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
import inspect

_CONFIG_FOR_DOC = "LlamaConfig"


def apply_rotary_pos_emb_1(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    # return x
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class LlamaFlashAttention_SQuat(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.cuda_bmm_fA_qB_outer = cuda_bmm_fA_qB_outer_rope if getattr(self.config, "cuda_bmm_implementation", "cos_sin") == "rope" else cuda_bmm_fA_qB_outer_cos_sin

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_aux_states: bool = False,
        save_aux_states: bool = False,
        aux_states: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]  #!!!
        
        # assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        feat_per_int = 32 // self.k_bits

        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]
            position_ids_past = past_key_value[8]
            Ainv_t = past_key_value[9]
            P_inv = past_key_value[10]

            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states = apply_rotary_pos_emb_1(query_states, cos, sin, position_ids) # !!!

            if key_states_quant_trans is not None:
                #=========================================================
                key_states_length = key_states_quant_trans.shape[-1] * feat_per_int
                position_ids_quant = position_ids_past[:,:key_states_length]
                cos, sin = self.rotary_emb(value_states, position_ids_quant)
                att_qkquant = self.cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                                                           key_scale_trans, key_mn_trans, cos, sin, self.k_bits)
                #=========================================================
            else:
                att_qkquant = None
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            
            #=========================================================
            position_ids = torch.cat((position_ids_past, position_ids), 1)  # !! remember here we already updated position_ids
            position_ids_full = position_ids[:,-key_states_full.shape[2]:]
            cos, sin = self.rotary_emb(value_states, position_ids_full)
            key_states_full_rot = apply_rotary_pos_emb_1(key_states_full, cos, sin, position_ids_full) # !!!
            #=========================================================

            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full_rot, self.num_key_value_groups).transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                # =========================================================
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = squat_lagrangian(key_states_full.transpose(2, 3).contiguous(), self.config.quant_group_size, self.group_size, self.k_bits, Ainv_t, P_inv)
                # =========================================================
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
            else:
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                                value_scale, value_mn, self.v_bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, self.num_key_value_groups))
            attn_output = attn_output.transpose(1, 2).contiguous()
            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn
            
            # position_ids = torch.cat((position_ids_past, position_ids), 1)

        else:
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                # Handle the case where the model is quantized
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states_rot = apply_rotary_pos_emb_1(query_states, cos, sin, position_ids) # !!!
            key_states_rot = apply_rotary_pos_emb_1(key_states, cos, sin, position_ids) # !!!

            if getattr(self.config, 'squat_query_subspace_prerope', 'after') == 'after':  # NOTE: 'before' or 'after'
                query_states = query_states_rot

            if getattr(self.config, "force_use_flash", False):
                # NOTE: only works with batch size = 1 (attention_mask is set to None)
                attn_output = self._flash_attention_forward(
                    query_states_rot.transpose(1, 2), key_states_rot.transpose(1, 2), 
                    value_states.transpose(1, 2), None, q_len, dropout=0.0
                )
            else:
                attn_output = self.naive_attention_forward(
                    query_states_rot, key_states_rot, 
                    value_states, attention_mask, q_len, dropout=0.0
                )

            # [b, nh, t, d] -> [b, nh/4, 4*t, d]
            if use_aux_states:
                # load auxiliary states
                layer_idx = kwargs['layer_idx']
                Ainv_t = aux_states[layer_idx]['Ainv_t']
                P_inv = aux_states[layer_idx]['P_inv']
            else:
                kv_nh = key_states.shape[1]
                head_dim = query_states.shape[3]
                subspace_dim = min(self.config.subspace_dim, self.num_key_value_groups*key_states.shape[2])

                # Get valid tokens from attention mask
                if attention_mask is not None:
                    # Get last row of attention mask [bs, 1, seq_len]
                    last_row_mask = attention_mask[:, :, -1, :]
                    # Find valid token positions (where mask is 0)
                    valid_tokens = (last_row_mask == 0).squeeze(1)  # [bs, seq_len]
                    
                    # Only keep valid tokens for each batch
                    query_subspace = []
                    for b in range(bsz):
                        # Get valid tokens for this batch
                        batch_valid = valid_tokens[b]  # [seq_len]
                        # Select valid tokens from query states
                        batch_query = query_states[b]  # [kv_nh, seq_len, head_dim] 
                        batch_valid_query = batch_query[:, batch_valid, :]  # [kv_nh, valid_len, head_dim]

                        valid_query_states_matrix = batch_valid_query.reshape(kv_nh, -1, head_dim)
                        if getattr(self.config, "power_method", False):
                            S_subspace, Vh_subspace = block_power_iteration(valid_query_states_matrix.float(), subspace_dim)
                            S_subspace = torch.diag_embed(S_subspace)
                        else:
                            U, S, Vh = torch.linalg.svd(valid_query_states_matrix.float(), full_matrices=False)
                            S_subspace = torch.diag_embed(S[:, :subspace_dim]).to(valid_query_states_matrix.dtype)
                            Vh_subspace = Vh[:, :subspace_dim, :].to(valid_query_states_matrix.dtype)
                        batch_query_subspace = torch.matmul(S_subspace, Vh_subspace)
                        query_subspace.append(batch_query_subspace)
                        if self.config.shared_svd == 'true':
                            break
                    
                    # Stack back into tensor
                    query_subspace = torch.stack(query_subspace)  # [bs, kv_nh, valid_len, head_dim]
                else:
                    # exit()
                    query_states_matrix = query_states.reshape(bsz, kv_nh, -1, head_dim)
                    U, S, Vh = torch.linalg.svd(query_states_matrix.float(), full_matrices=False)  #!!! float here might be suboptimal
                    S_subspace = torch.diag_embed(S[:, :, :subspace_dim]).to(query_states_matrix.dtype)
                    Vh_subspace = Vh[:, :, :subspace_dim, :].to(query_states_matrix.dtype)

                    # dimension: [bs, nh, subspace_dim, head_dim]
                    query_subspace = torch.matmul(S_subspace, Vh_subspace)

                if self.config.shared_svd == 'true':
                    query_subspace = query_subspace[0:1, ...]

                # Ainv_t is a list of  matrices
                Ainv_t = generate_At_inv(self.config.quant_group_size, query_subspace.float(), lamb = self.config.squat_lambda)
                P_inv = torch.inverse(Ainv_t[-1])

                if save_aux_states:
                    layer_idx = kwargs['layer_idx']
                    aux_states[layer_idx] = {
                        'Ainv_t': Ainv_t,
                        'P_inv': P_inv,
                    }

            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:

                # key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
                if self.config.fill_key_quant_with_first_col == "true":
                    quant_seq_len = key_states_quant.shape[-2]
                    attn_mask = torch.cat([torch.zeros((attention_mask.shape[0], 1, 1), dtype=torch.bool, device=attention_mask.device), attention_mask[:,:,-1,:quant_seq_len] == 0], dim=2)  # [bs, 1, len]
                    is_first_token = torch.bitwise_xor(attn_mask[:,:,1:], attn_mask[:,:,:-1]).unsqueeze(-1)
                    key_states_quant_first_col = key_states_quant[is_first_token.expand(-1, key_states_quant.shape[1], -1, key_states_quant.shape[-1])].reshape(bsz, key_states_quant.shape[1], 1, key_states_quant.shape[-1])
                    key_states_quant_fixed = torch.where(attn_mask[:,:,1:].unsqueeze(-1), key_states_quant, key_states_quant_first_col)
                else:
                    key_states_quant_fixed = key_states_quant
                key_states_quant_trans, key_scale_trans, key_mn_trans = squat_lagrangian(key_states_quant_fixed.transpose(2, 3).contiguous(), self.config.quant_group_size, self.group_size, self.k_bits, Ainv_t, P_inv)

            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)

        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, 
                          value_states_quant, value_states_full, value_scale, value_mn, position_ids, Ainv_t, P_inv, kv_seq_len) if use_cache else None
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value
    
    def naive_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        kv_seq_len = key_states.shape[-2]
        bsz, nh, q_len, hd = query_states.shape

        attn_weights = torch.matmul(query_states, repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
        
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, repeat_kv(value_states, self.num_key_value_groups))
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

   
class LlamaFlashAttention_SQuat_postrope(LlamaFlashAttention_SQuat):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_aux_states: bool = False,
        save_aux_states: bool = False,
        aux_states: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]  #!!!
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]
            position_ids_past = past_key_value[8]
            Ainv_t = past_key_value[9]
            P_inv = past_key_value[10]
            key_states_pre = past_key_value[11]
            value_states_pre = past_key_value[12]
            if key_states_pre is not None:
                att_qkpre = torch.matmul(query_states, repeat_kv(key_states_pre, self.num_key_value_groups).transpose(2, 3))
            else:
                att_qkpre = None
            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                                key_scale_trans, key_mn_trans, self.k_bits)
            else:
                att_qkquant = None
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))
            if att_qkpre is not None:
                attn_weights = att_qkpre
            else:
                attn_weights = torch.zeros((bsz, self.num_heads, q_len, 0), device=query_states.device)
            if att_qkquant is not None:
                attn_weights = torch.cat([attn_weights, att_qkquant], dim=-1)
            attn_weights = torch.cat([attn_weights, att_qkfull], dim=-1) / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                # =========================================================
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = squat_lagrangian(key_states_full.transpose(2, 3).contiguous(), self.config.quant_group_size, self.group_size, self.k_bits, Ainv_t, P_inv)
                # =========================================================
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            if value_states_pre is not None:
                value_pre_length = value_states_pre.shape[-2]
                attn_output = torch.matmul(attn_weights[:, :, :, :value_pre_length], repeat_kv(value_states_pre, self.num_key_value_groups))
            else:
                value_pre_length = 0
                attn_output = 0.
            if value_states_full is not None:
                value_states_full = torch.cat([value_states_full, value_states], dim=2)
            else:
                value_states_full = value_states
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                attn_output += torch.matmul(attn_weights[:, :, :, value_pre_length:], repeat_kv(value_states_full, self.num_key_value_groups))
            else:
                attn_output += cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, value_pre_length:-value_full_length], value_states_quant, 
                                                value_scale, value_mn, self.v_bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, self.num_key_value_groups))
            attn_output = attn_output.transpose(1, 2).contiguous()
            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

        else:
            residual_length = self.config.residual_length
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                # Handle the case where the model is quantized
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            if getattr(self.config, "force_use_flash", False):
                attn_output = self._flash_attention_forward(
                    query_states.transpose(1, 2), key_states.transpose(1, 2), 
                    value_states.transpose(1, 2), None, q_len, dropout=0.0
                )
            else:
                attn_output = self.naive_attention_forward(
                    query_states, key_states, 
                    value_states, attention_mask, q_len, dropout=0.0
                )

            # [b, nh, t, d] -> [b, nh/4, 4*t, d]
            if use_aux_states:
                # load auxiliary states
                layer_idx = kwargs['layer_idx']
                Ainv_t = aux_states[layer_idx]['Ainv_t']
                P_inv = aux_states[layer_idx]['P_inv']
            else:
                kv_nh = key_states.shape[1]
                head_dim = query_states.shape[3]
                subspace_dim = min(self.config.subspace_dim, self.num_key_value_groups*key_states.shape[2])

                # Get valid tokens from attention mask
                if attention_mask is not None:
                    # Get last row of attention mask [bs, 1, seq_len]
                    last_row_mask = attention_mask[:, :, -1, :]
                    # Find valid token positions (where mask is 0)
                    valid_tokens = (last_row_mask == 0).squeeze(1)  # [bs, seq_len]
                    
                    # Only keep valid tokens for each batch
                    query_subspace = []
                    for b in range(bsz):
                        # Get valid tokens for this batch
                        batch_valid = valid_tokens[b]  # [seq_len]
                        # Select valid tokens from query states
                        batch_query = query_states[b]  # [kv_nh, seq_len, head_dim] 
                        batch_valid_query = batch_query[:, batch_valid, :]  # [kv_nh, valid_len, head_dim]

                        valid_query_states_matrix = batch_valid_query.reshape(kv_nh, -1, head_dim)

                        if getattr(self.config, "power_method", False):
                            S_subspace, Vh_subspace = block_power_iteration(valid_query_states_matrix.float(), subspace_dim)
                            S_subspace = torch.diag_embed(S_subspace)
                        else:
                            U, S, Vh = torch.linalg.svd(valid_query_states_matrix.float(), full_matrices=False)
                            S_subspace = torch.diag_embed(S[:, :subspace_dim]).to(valid_query_states_matrix.dtype)
                            Vh_subspace = Vh[:, :subspace_dim, :].to(valid_query_states_matrix.dtype)
                        batch_query_subspace = torch.matmul(S_subspace, Vh_subspace)

                        # S_subspace, Vh_subspace = block_power_iteration(valid_query_states_matrix.float(), subspace_dim)
                        # Compute batch_query_subspace (reconstruct using singular values)
                        # batch_query_subspace = torch.matmul(torch.diag_embed(S_subspace), Vh_subspace)

                        query_subspace.append(batch_query_subspace)
                        if self.config.shared_svd == 'true':
                            break
                    
                    # Stack back into tensor
                    query_subspace = torch.stack(query_subspace)  # [bs, kv_nh, valid_len, head_dim]
                else:
                    # exit(0)
                    query_states_matrix = query_states.reshape(bsz, kv_nh, -1, head_dim)
                    U, S, Vh = torch.linalg.svd(query_states_matrix.float(), full_matrices=False)  #!!! float here might be suboptimal
                    S_subspace = torch.diag_embed(S[:, :, :subspace_dim]).to(query_states_matrix.dtype)
                    Vh_subspace = Vh[:, :, :subspace_dim, :].to(query_states_matrix.dtype)

                    # dimension: [bs, nh, subspace_dim, head_dim]
                    query_subspace = torch.matmul(S_subspace, Vh_subspace)

                if self.config.shared_svd == 'true':
                    query_subspace = query_subspace[0:1, ...]

                # Ainv_t is a list of  matrices
                Ainv_t = generate_At_inv(self.config.quant_group_size, query_subspace.float(), lamb = self.config.squat_lambda)
                P_inv = torch.inverse(Ainv_t[-1])

                if save_aux_states:
                    layer_idx = kwargs['layer_idx']
                    aux_states[layer_idx] = {
                        'Ainv_t': Ainv_t,
                        'P_inv': P_inv,
                    }
            
            # quantize
            if key_states.shape[-2] % residual_length != 0:
                if key_states.shape[-2] < residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                if self.config.fill_key_quant_with_first_col == "true":
                    quant_seq_len = key_states_quant.shape[-2]
                    attn_mask = torch.cat([torch.zeros((attention_mask.shape[0], 1, 1), dtype=torch.bool, device=attention_mask.device), attention_mask[:,:,-1,:quant_seq_len] == 0], dim=2)  # [bs, 1, len]
                    is_first_token = torch.bitwise_xor(attn_mask[:,:,1:], attn_mask[:,:,:-1]).unsqueeze(-1)
                    key_states_quant_first_col = key_states_quant[is_first_token.expand(-1, key_states_quant.shape[1], -1, key_states_quant.shape[-1])].reshape(bsz, key_states_quant.shape[1], 1, key_states_quant.shape[-1])
                    key_states_quant_fixed = torch.where(attn_mask[:,:,1:].unsqueeze(-1), key_states_quant, key_states_quant_first_col)
                else:
                    key_states_quant_fixed = key_states_quant
                key_states_quant_trans, key_scale_trans, key_mn_trans = squat_lagrangian(key_states_quant_fixed.transpose(2, 3).contiguous(), self.config.quant_group_size, self.group_size, self.k_bits, Ainv_t, P_inv)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-residual_length, :].contiguous()
                value_states_full = value_states[:, :, -residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
            
            key_states_pre = None
            value_states_pre = None

        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, 
                          value_states_quant, value_states_full, value_scale, value_mn, position_ids, 
                          Ainv_t, P_inv, key_states_pre, value_states_pre,kv_seq_len) if use_cache else None
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer_SQuat(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        assert getattr(config, "use_flash", False)
        if config.method.lower() in ["squat", "squat_prerope", "squat_pre"]:
            self.self_attn = LlamaFlashAttention_SQuat(config=config)
        elif config.method.lower() in ["squat_postrope", "squat_post"]:
            self.self_attn = LlamaFlashAttention_SQuat_postrope(config=config)
        else:
            raise ValueError(f"Invalid method: {config.method}")
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel_SQuat(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_SQuat(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    layer_idx=idx,
                    **kwargs
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_SQuat(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_SQuat(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = past_key_values.to_legacy_cache()
            if len(past_key_values) == 0:
                past_key_values = None
        if past_key_values is not None:
            past_length = past_key_values[0][-1]  #!!!
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        if 'aux_states' in kwargs:
            model_inputs['use_aux_states'] = True
            model_inputs['aux_states'] = kwargs['aux_states']
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]) -> None:
        """Override the validation to allow our custom kwargs"""
        # Remove our custom kwargs before validation
        use_aux_states = model_kwargs.pop("use_aux_states", None)
        save_aux_states = model_kwargs.pop("save_aux_states", None)
        aux_states = model_kwargs.pop("aux_states", None)
        
        # Call parent validation for remaining kwargs
        super()._validate_model_kwargs(model_kwargs)
        
        # Put our kwargs back
        if save_aux_states is not None:
            model_kwargs["save_aux_states"] = save_aux_states
        if use_aux_states is not None:
            model_kwargs["use_aux_states"] = use_aux_states
        if aux_states is not None:
            model_kwargs["aux_states"] = aux_states

    def save_auxiliary_states(
        self,
        inputs: Optional[torch.Tensor] = None,
        save_aux_states: bool = True,
        use_aux_states: bool = False,
        aux_states: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config, model_kwargs = self._prepare_generation_config(None, **kwargs)

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"

        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory

        past = model_kwargs.get(cache_name, None)
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )
        if past is None:
            model_kwargs[cache_name] = (
                DynamicCache()
                if not requires_cross_attention_cache
                else EncoderDecoderCache(DynamicCache(), DynamicCache())
            )

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs, 
                       aux_states=aux_states, 
                       save_aux_states=save_aux_states,
                       use_aux_states=use_aux_states,
                       return_dict=True)
        return outputs

    @torch.no_grad()
    def generate2(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_aux_states: bool = True,
        aux_states: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if aux_states is not None:
            print(f"<< Generating with auxiliary states >>")

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        # self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        # if synced_gpus is None:
        #     if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
        #         synced_gpus = True
        #     else:
        #         synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        # if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        #     # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        #     # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        #     if (
        #         generation_config._pad_token_tensor is not None
        #         and batch_size > 1
        #         and len(inputs_tensor.shape) == 2
        #         and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        #     ):
        #         logger.warning(
        #             "A decoder-only architecture is being used, but right-padding was detected! For correct "
        #             "generation results, please set `padding_side='left'` when initializing the tokenizer."
        #         )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        use_dynamic_cache_by_default = False
        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"

        if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
            raise ValueError(
                f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs[cache_name] = cache_class(cache_config)
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            requires_cross_attention_cache = (
                self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                model_kwargs[cache_name] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )
                use_dynamic_cache_by_default = True

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # 10. go into different generation modes
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                use_aux_states=use_aux_states,
                aux_states=aux_states,
                **model_kwargs,
            )

        # Convert to legacy cache if needed
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
                if isinstance(result.past_key_values, (DynamicCache, EncoderDecoderCache)):
                    result.past_key_values = result.past_key_values.to_legacy_cache()
        return result
