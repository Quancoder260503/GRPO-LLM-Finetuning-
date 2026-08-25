import json 
import dataclasses 
from pathlib import Path 
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F 
from torch import nn 
import safetensors.torch

@dataclasses.dataclass
class QwenConfig:  
  attention_dropout : float = 0.2 
  bos_token_id : int = 151643 
  eos_token_id : int = 151645
  hidden_act : str = 'silu' 
  hidden_size : int =  2048 
  initializer_range : float = 0.02 
  intermediate_size : int = 11008 
  max_position_embedding : int = 32768 
  max_window_layers : int = 70 
  model_type : str = "qwen2" 
  num_attention_heads : int = 16 
  num_hidden_layers : int = 36 
  num_key_value_heads : int = 2 
  rms_norm_eps : float = 1e-06
  rotate_theta : float = 1000000.0 
  sliding_window : int = 32768
  tie_word_embedding : bool = True
  torch_dtype : str = "bfloat16" 
  use_cache : bool = True 
  use_sliding_window : bool = True 
  vocab_size : int = 151936 

class RMSNorm(nn.Module): 
  def __init__(self, dim : int, eps : float = 1e-6):
    super(RMSNorm, self).__init__() 
    self.eps = eps 
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x): 
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) # inverse square root 

  def forward(self, x): 
    input_dtype = x.dtype 
    x = x.to(torch.float32)
    x = self._norm(x).type_as(x) 
    x = self.weight * x.to(input_dtype)
    return x 

def rotate_half(x): 
  x1 = x[..., : x.shape[-1] // 2] 
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat([x2, x1], dim = -1) 

def apply_rotate_positional_embedding(q, k, cos, sin, unsqueeze_dim = 2): 
  cos = cos.unsqueeze(unsqueeze_dim) 
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin) 
  k_embed = (k * cos) + (rotate_half(k) * sin) 
  return q_embed, k_embed 

class Attention(nn.Module):
  def __init__(self, args):
    super(Attention, self).__init__() 
    self.n_kv_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads 
    self.n_heads = args.num_attention_heads
    self.n_kv_heads = self.n_kv_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.head_dim = args.hidden_size // args.num_attention_heads
    
    self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias = True)
    self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias = True)
    self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias = True)
    self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias = False)
    self.args = args

  def init_kv_cache(self, max_batch_size : int, max_seq_len : int, dtype : torch.dtype, device = torch.device):
    cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
    cache_k = torch.zeros(cache_shape, dtype = dtype, device = device) 
    cache_v = torch.zeros(cache_shape, dtype = dtype, device = device) 
    self.register_buffer("cache_k", cache_k, persistent = False) 
    self.register_buffer("cache_v", cache_v, persistent = False) 

  def del_kv_cache(self): 
    self.cache_k = None 
    self.cache_v = None 

  def forward(
    self, 
    x : torch.Tensor, 
    positional_embedding : Tuple[torch.Tensor, torch.Tensor], 
    start_position : Optional[Union[int, torch.Tensor]] = None, 
  ): 
    batch_size, seq_len, _ = x.shape 
    q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x) 
    q = q.view(batch_size, seq_len, self.args.num_attention_heads, self.head_dim)
    k = k.view(batch_size, seq_len, self.args.num_attention_heads, self.head_dim)
    v = v.view(batch_size, seq_len, self.args.num_attention_heads, self.head_dim)

    cos, sin = positional_embedding 
    q, k = apply_rotate_positional_embedding(q, k, cos, sin) 

    if start_position is not None: 
      end_position = start_position + seq_len 
      self.cache_k[:batch_size, start_position : end_position, :, :] = k
      self.cache_v[:batch_size, start_position : end_position, :, :] = v 
      output = torch.nn.functional.scaled_dot_product_attention(
        query = q.transpose(1, 2), 
        key = self.cache_k[:batch_size, :end_position].transpose(1, 2), 
        value = self.cache_v[:, batch_size, :end_position].transpose(1, 2), 
        is_causal = True if seq_len > 1 else False, 
        enable_gqa = True, 
      ).transpose(1, 2)
    else : 
      output = torch.nn.functional.scaled_dot_product_attention(
        query = q.transpose(1, 2), 
        key = k.transpose(1, 2), 
        value = v.transpose(1, 2), 
        is_causal = True, 
        enable_gqa = True, 
      ).transpose(1, 2) 
    output = output.reshape(batch_size, seq_len, -1) 
    output = self.o_proj(output)
    return output 


class FeedForward(nn.Module): 
  def __init__(self, dim : int, intermediate_size : int):
    super(FeedForward, self).__init__() 
    self.up_proj = nn.Linear(dim, intermediate_size,   bias = False) 
    self.gate_proj = nn.Linear(dim, intermediate_size, bias = False) 
    self.down_proj = nn.Linear(intermediate_size, dim, bias = False)


  def forward(self, x):
    x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    return x 


class TransformerBlock(nn.Module): 
  def __init__(self, layer_id : int, args):
    super(TransformerBlock, self).__init__() 
    self.n_heads = args.num_attention_heads 
    self.dim = args.hidden_size 
    self.head_dim = args.hidden_size // args.num_attention_heads 
    self.self_attn = Attention(args) 
    self.mlp = FeedForward(
      dim = args.hidden_size, 
      intermediate_size = args.intermediate_size, 
    )
    self.layer_id = layer_id 
    self.input_layernorm = RMSNorm(args.hidden_size, eps = args.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(args.hidden_size, eps = args.rms_norm_eps) 

  def forward(
    self, 
    x : torch.Tensor, 
    positional_embedding : Tuple[torch.Tensor, torch.Tensor], 
    start_position : Optional[Union[int, torch.Tensor]] = None, 
  ): 
    h = x + self.self_attn(self.input_layernorm(x), positional_embedding, start_position = start_position)
    out = h + self.mlp(self.post_attention_layernorm(h))
    return out 


class Qwen2RotaryEmbedding(nn.Module): 
  def __init__(self, config, device):
   super(Qwen2RotaryEmbedding, self).__init__()
   self.config = config 
   base = config.rotate_theta
   dim = config.hidden_size // config.num_attention_heads 

   with torch.autocast(device_type = device.type, dtype = torch.float32): 
     inv_freq = 1.0 / (base ** torch.arange(0, dim, 2, dtype = torch.int64).float().to(device) / dim)

   self.register_buffer('inv_freq', inv_freq, persistent = False) 

  @torch.no_grad() 
  def forward(self, x, pos): 
    inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
    pos = pos[:, None, :].float()
    device_type = x.device.type 
    with torch.autocast(device_type = device_type, enabled = False): 
      freqs = (inv_freq.float().to(x.device) @ pos.float()).transpose(1, 2)
      embed = torch.cat((freqs, freqs), dim = -1) 
      cos = embed.cos()
      sin = embed.sin() 
    return cos.to(dtype = x.dtype), sin.to(dtype = x.dtype) 


class Transformer(nn.Module): 
  def __init__(self, params, device):
    super(Transformer, self).__init__() 
    self.params = params
    self.vocab_size = params.vocab_size 
    self.n_layers = params.num_hidden_layers 

    self.embed_tokens = nn.Embedding(params.vocab_size, params.hidden_size)

    with torch.device(device): 
      self.rotary_emb = Qwen2RotaryEmbedding(config = params, device = device) 

    self.layers = torch.nn.ModuleList()

    for layer_id in range(params.num_hidden_layers): 
      self.layers.append(TransformerBlock(layer_id, params)) 

    self.norm = RMSNorm(dim = params.hidden_size, eps = params.rms_norm_eps) 
  
    if not params.tie_word_embedding: 
      self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias = False) 
    


  def output_proj(self, x): 
    if self.params.tie_word_embeddings: 
      return x @ self.embed_tokens.weight.T 
    return self.lm_head(x) 

  def forward(self, tokens : torch.Tensor):
    batch_size, seq_len, _ = tokens.shape 
    h = self.embed_tokens(tokens)
    pos = torch.arange(0, seq_len, device = tokens.device, dtype = torch.int32)
    pos_emb = self.rotary_emb(h, pos[None, :])

    pipe = [] 
    for layer in self.layers:
      pipe.append(lambda x, layer : layer(x, pos_emb))

    pipe.append(self.norm.forward)
    pipe.append(self.output_proj) 
    return torch.utils.checkpoint.checkpoint_sequential(
      pipe, 
      len(pipe), 
      h, 
      use_reentrant = False, 
    )

  def inference(self, tokens : torch.Tensor, start_pos : Union[int, torch.Tensor]): 
    _, seq_len = tokens.shape 
    h = self.embed_tokens(tokens)

    pos = torch.arange(0, seq_len, device = tokens.device, dtype = torch.int32)[None,:]
    if isinstance(start_pos, torch.Tensor): 
      pos = pos + start_pos[:, None] 
    else : 
      pos.add_(start_pos)

    pos_emb = self.rotary_emb(h, pos)

    for layer in self.layers:
      h = layer(h, pos_emb, start_pos = start_pos)

    h = h[:, -1:, :]
    h = self.norm(h) 
    output = self.output_proj(h)
    return output 


  def init_kv_cache(self, max_batch_size : int, max_seq_len : int, device : torch.device, dtype = torch.dtype):
    for layer in self.layers:
      layer.self_attn.init_kv_cache(
        max_batch_size = max_batch_size, 
        max_seq_len = max_seq_len, 
        device = device, 
        dtype = dtype 
      )

  def del_kv_cache(self): 
    for layer in self.layers: 
      layer.self_attn.del_kv_cache() 


  @classmethod 
  def from_pretrained(cls, checkpoint_path, device = torch.device): 
    config_file = Path(checkpoint_path) / "config.json" 
    with open(config_file, "r") as f: 
      config = json.load(f)
    args = QwenConfig(
       attention_dropout = config['attention_dropout'],
       bos_token_id = config['bos_token_id'],
       eos_token_id = config['eos_token_id'],
       hidden_act = config['hidden_act'],
       hidden_size = config['hidden_size'],
       initializer_range = config['initializer_range'],
       intermediate_size = config['intermediate_size'],
       max_position_embedding = config['max_position_embeddings'],
       max_window_layers = config['max_window_layers'],
       model_type = config['model_type'],
       num_attention_heads = config['num_attention_heads'],
       num_hidden_layers = config['num_hidden_layers'],
       num_key_value_heads = config['num_key_value_heads'],
       rms_norm_eps = config['rms_norm_eps'],
       rotate_theta = config['rope_theta'],
       sliding_window = config['sliding_window'],
       tie_word_embedding = config['tie_word_embeddings'],
       torch_dtype = config['torch_dtype'],
       use_cache = config['use_cache'],
       use_sliding_window = config['use_sliding_window'],
       vocab_size = config['vocab_size'],
    )
    with torch.device('meta'): 
      model = cls(params = args, device = device) 

    model_weight_files = sorted(Path(checkpoint_path).glob("model*.safetensors"))
    weights = {} 
    for file in model_weight_files: 
      weights.update(safetensors.torch.load_file(file, device = 'cpu'))

    weights = {k.replace("model.", "") : v for k, v in weights.items()}
    model.load_state_dict(weights, strict = True, assign = True) 
    return model.to(device)