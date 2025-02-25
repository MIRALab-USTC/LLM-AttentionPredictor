# coding=utf-8

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, StaticCache
from transformers.utils import (
    logging,
    is_flash_attn_greater_or_equal_2_10,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, LlamaFlashAttention2, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import _flash_attention_forward

__all__ = ['LlamaFlashAttention2_AttnPred', 'convert_kvcache_llama_attnpred']

logger = logging.get_logger(__name__)


class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)  

    def forward(self, x):
        # x: (batch_size, a, b)
        x = x.unsqueeze(1) # x: (batch_size, 1, a, b)
        x = self.relu(self.conv1(x)) # (batch_size, 16, a, b)
        x = self.relu(self.conv2(x)) # (batch_size, 32, a, b)
        x = self.pool(x) # (batch_size, 64, 1, b)
        x = x.view(x.size(0), x.size(1), -1)  # (batch_size, 64, b)
        x = self.conv3(x)  # (batch_size, 1, b)
        x = x.squeeze(1)  # (batch_size, b)
        return x    

class LlamaFlashAttention2_AttnPred(nn.Module):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super(LlamaFlashAttention2_AttnPred, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


        ### AttentionPredictor
        self.model = CNN()
        # self.model.load_state_dict(torch.load('/home/qyyang/kvcache/test/demo/runs/2024-10-25_09-23-56_CNN_longchat_lb_64_pool16_alltask_5case/best_model.pth', map_location='cuda:0'))
        # self.model.load_state_dict(torch.load('/home/qyyang/kvcache/test/demo/runs/2024-11-02_21-53-46_CNN_llama3.1_alltask_5case/best_model.pth', map_location='cuda:0'))       

        self.attn_history = None
        self.topk = 1024
        self.time_flag = True
        self.sink_token = 64
        self.local_token = 64
        self.history_step = 64
        self.pooling_block_size = 16
        self.calibration_step = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if q_len==1: # decode don't use flash attention
            self.decode_length = self.decode_length+1
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            ### start AttentionPredictor

            attn_weights_full = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            tsp_attn = self.time_series_predict(self.attn_history, token_id=key_states.shape[-2])
            tsp_mask = self.create_tsp_mask(attn_weights_full, tsp_attn, topk=self.topk)
            # self.recovery_rate = torch.sum(attn_weights_full[torch.where(tsp_mask==0)])/torch.sum(attn_weights_full)
            attn_weights = attn_weights + tsp_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            if self.calibration_step is not None and self.decode_length % self.calibration_step == 0:
                self.attn_history = self.update_attn_history(self.attn_history, attn_weights_full)
            else:
                self.attn_history = self.update_attn_history(self.attn_history, attn_weights)

            ### end AttentionPredictor

            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2)
        
        else: # prefill use flash attention
            self.decode_length = 0
            
            ### start AttentionPredictor
            # calculate the last H tokens' attention for attention history
            key_states_for_history = repeat_kv(key_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states[:,:,-self.history_step:,:], key_states_for_history.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states_for_history.shape[-2]]
                attn_weights = attn_weights + causal_mask

            self.attn_history = None
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_weights: [bsz, head_num, q_len=64, attn_len]
            self.attn_history = self.update_attn_history(self.attn_history, attn_weights)
            ### end AttentionPredictor

            ### Flash Attention
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (LlamaRMSNorm handles it correctly)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
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

            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )

        

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def update_attn_history(self, attn_history, attn_weights_full):
        if attn_history is None: # initialize attn_history
            attn_pooling = self.max_pooling(attn_weights_full, self.pooling_block_size)
            attn_history = attn_pooling[:, :, -self.history_step:, :]
        else:
            # attn_history: [bsz, head_num, 64, attn_len]
            # update attn_history 
            attn_pooling = self.max_pooling(attn_weights_full, self.pooling_block_size)
            # after pooling, attn_pooling: [bsz, head_num, 1, attn_len/block_size]
            if attn_pooling.shape[-1] == attn_history.shape[-1]:
                attn_history = torch.cat([attn_history[:, :, 1:, :], attn_pooling], dim=-2)
            else:
                zero_padding = torch.zeros(attn_history.size(0), attn_history.size(1), attn_history.size(2)-1, 1, device=attn_history.device, dtype=attn_history.dtype)
                attn_history = torch.cat([attn_history[:, :, 1:, :], zero_padding], dim=-1)
                attn_history = torch.cat([attn_history, attn_pooling], dim=-2)
        return attn_history
    
    def max_pooling(self, tensor, block_size):
        # padding tensor with zeros
        padding_size = (block_size - tensor.shape[-1] % block_size) % block_size
        padded_tensor = torch.cat([tensor, torch.zeros(tensor.shape[:-1] + (padding_size,), dtype=tensor.dtype, device=tensor.device)],dim=-1)
        # max pooling
        pooled_tensor = padded_tensor.view(*padded_tensor.shape[:-1], -1, block_size).max(dim=-1)[0]
        return pooled_tensor
    
    def time_series_predict(self, attn_history, token_id):
        # attn_history: [bsz, head_num, 64, attn_len']
        bsz, head_num, num_rows, attn_len = attn_history.shape
        start = self.sink_token//self.pooling_block_size
        end = attn_len-(self.local_token//self.pooling_block_size)
        attn_history = attn_history[:, :, :, start:end]
        attn_len = attn_history.shape[-1]

        if end-start < 3: # CNN kernel is 3*3 size
            tsp_attn = torch.ones(bsz, head_num, 1, attn_len).to(attn_history.device)
            return tsp_attn
        
        with torch.no_grad():
            inputs = attn_history.view(bsz * head_num, num_rows, attn_len) # predict all heads together
            tsp_attn = self.model(inputs)
        tsp_attn = tsp_attn.view(bsz, head_num, 1, attn_len)
        return tsp_attn

    def create_tsp_mask(self, attn_weights, tsp_attn, topk=5):
        # drop positions=-10000, keep positions=0, attention_weights=mask+attention_weights
        tsp_mask = torch.full_like(attn_weights, -10000.0).to(attn_weights.dtype)  # initialize with -10000

        if tsp_attn.shape[-1]<1:
            tsp_mask[:, :, :, :self.sink_token] = 0
            tsp_mask[:, :, :, -self.local_token:] = 0
            return tsp_mask

        # get topk indices
        _, topk_indices = torch.topk(tsp_attn, min((topk-self.sink_token-self.local_token)//self.pooling_block_size,tsp_attn.shape[-1]), dim=-1)

        # expand block indices [batch_size, topk] to attention indices [batch_size, topk * self.pooling_block_size] 0->0~15
        topk_indices = topk_indices.unsqueeze(-1) * self.pooling_block_size + torch.arange(self.pooling_block_size).to(topk_indices.device)
        topk_indices = topk_indices.view(*topk_indices.shape[:-2], -1)

        # set topk indices to 0
        tsp_mask.scatter_(-1, topk_indices + self.sink_token, 0)
        
        # set sink_token and local_token to 0
        tsp_mask[:, :, :, :self.sink_token] = 0
        tsp_mask[:, :, :, -self.local_token:] = 0

        return tsp_mask


def convert_kvcache_llama_attnpred(model, model_type='llama-3.1', topk = 1024, skip_2layer=False, type = None, sink_token = 64, local_token = 64, calibration_step=5):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_attnpred(module, topk=topk, skip_2layer=skip_2layer, type = type, sink_token = sink_token, local_token = local_token, calibration_step=calibration_step)

        if isinstance(module, LlamaAttention):
            original_config = module.config
            layer_idx = module.layer_idx
            if skip_2layer and (layer_idx == 0 or layer_idx == 1):
                continue
            param_dtype = next(module.parameters()).dtype

            if module.__class__ == LlamaFlashAttention2:
                new_module = LlamaFlashAttention2_AttnPred(original_config, layer_idx)
                if 'llama-3.1' in model_type.lower :
                    new_module.model.load_state_dict(torch.load('../../model/CNN_llama3.1_alltask_5case/best_model.pth', map_location='cuda:0'))
                elif 'longchat' in model_type.lower:
                    new_module.model.load_state_dict(torch.load('../../model/CNN_longchat_alltask_5case/best_model.pth', map_location='cuda:0'))
                else:
                    Exception('model_type error')

                new_module.calibration_step = calibration_step
                new_module.topk = topk
                new_module.sink_token = sink_token
                new_module.local_token = local_token

            for param in new_module.parameters():
                param.data = param.data.to(param_dtype)
            model._modules[name] = new_module
    return model
