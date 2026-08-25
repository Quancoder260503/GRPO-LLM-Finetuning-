# remember to paste the copyright here 
import math
from dataclasses import dataclass 
from typing import List, Optional, Tuple, TypedDict
from pathlib import Path
import torch 
import fairscale.nn.model_parallel.initialize as fs_init 
import torch.nn.functional as F 
from fairscale.nn.model_parallel.layers import ( 
  ColumnParallelLinear, 
  RowParallelLinear, 
  VocabParallelEmbedding, 
)
from torch import nn 

@dataclass 
class ModelArgs: 
  dim : int = 4096 
  n_layers : int = 32 
  n_heads : int = 32 
  n_kv_heads : Optional[int] = None 
  vocab_size : int = -1 
  multiple_of : int = 256 
  ffn_dim_multiplier : Optional[float] = None
  norm_eps : float = 1e-5 
  rope_theta : float = 500000
  max_batch_size : int = 32 
  max_seq_len : int = 2048 

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

def precompute_freqs_cis(dim : int, end : int, theta : float = 10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device = freqs.device, dtype = torch.float32) 
  freqs = torch.outer(t, freqs) 
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis

def reshape_for_broadcast(freq_cis : torch.Tensor, x : torch.Tensor):
  ndim = x.ndim 
  assert 0 <= 1 < ndim 
  assert freq_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freq_cis.view(*shape)

def apply_rotary_embed(
  q: torch.Tensor, 
  k : torch.Tensor, 
  freqs_cis : torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  xq = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
  xk = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq)
  xq_out = torch.view_as_real(xq * freqs_cis).flatten(3) 
  xk_out = torch.view_as_real(xk * freqs_cis).flatten(3) 
  return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x : torch.Tensor, n_rep : int) -> torch.Tensor: 
  batch_size, seq_len, num_kv_heads, head_dim = x.shape
  if n_rep == 1: 
    return x 
  return (
    x[:, :, :, None, :]
    .expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
    .reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
  )

class Attention(nn.Module): 
  def __init__(self, args : ModelArgs): 
    super(Attention, self).__init__() 
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads 
    model_parallel_size = fs_init.get_model_parallel_world_size()
    self.n_local_heads = args.n_heads // model_parallel_size
    self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
    self.n_rep = self.n_local_heads // self.n_local_kv_heads 
    self.head_dim = args.dim // args.n_heads

    self.wq = ColumnParallelLinear(
      in_features = args.dim, 
      out_features = args.n_heads * self.head_dim, 
      bias = False, 
      gather_output = False, 
      init_method = lambda x : x 
    )

    self.wk = ColumnParallelLinear(
      in_features = args.dim, 
      out_features = args.n_heads * self.head_dim, 
      bias = False, 
      gather_output = False, 
      init_method = lambda x : x 
    )

    self.wv = ColumnParallelLinear(
      in_features = args.dim, 
      out_features = args.n_heads * self.head_dim, 
      bias = False, 
      gather_output = False, 
      init_method = lambda x : x 
    )

    self.wo = ColumnParallelLinear(
      in_features = args.dim, 
      out_features = args.n_heads * self.head_dim, 
      bias = False, 
      gather_output = False, 
      init_method = lambda x : x 
    )

    self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda() 
    self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda()


  def forward(self, x : torch.Tensor, start_position : int, freqs_cis : torch.Tensor, mask : Optional[torch.Tensor]):
    batch_size, seq_len, _ = x.shape 
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) 

    xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim) 
    xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

    xq, xk = apply_rotary_embed(xq, xk, freqs_cis)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)

    self.cache_k[:batch_size, start_position : start_position + seq_len] = xk 
    self.cache_v[:batch_size, start_position : start_position + seq_len] = xv

    keys = self.cache_k[:batch_size, : (start_position + seq_len)]
    values = self.cache_v[:batch_size, : (start_position + seq_len)]

    keys = repeat_kv(keys, self.n_rep)  # [batch_size, cache_len + seq_len, n_local_heads, head_dim]
    values = repeat_kv(values, self.n_rep) # [batch_size, cache_len + seq_len, n_local_heads, head_dim]

    xq = xq.transpose(1, 2) # [batch_size, n_local_heads, seq_len, head_dim]
    keys = keys.transpose(1, 2) # [batch_size, n_local_heads, cache_len + seq_len, head_dim] 
    values = values.transpose(1, 2) # [batch_size, n_local_heads, cache_len + seq_len, head_dim] 

    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None: 
      scores = scores + mask 

    scores = F.softmax(scores.float(), dim = -1).type_as(xq)
    output = torch.matmul(scores, values) 
    output = output.transpose.contiguous().view(batch_size, seq_len, -1) 
    output = self.wo(output) 
    return output 

class FeedForward(nn.Module): 
  def __init__(self, dim : int, hidden_dim : int, multiple_of : int, ffn_dim_multiplier : Optional[float]): 
    super(FeedForward, self).__init__()
    hidden_dim = int(2 * hidden_dim / 3) 
    if ffn_dim_multiplier is not None: 
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = ColumnParallelLinear(dim, hidden_dim, bias = False, gather_output = False, init_method = lambda x : x) 
    self.w2 = RowParallelLinear(dim, hidden_dim, bias = False, input_is_parallel = True, init_method = lambda x : x) 
    self.w3 = ColumnParallelLinear(dim, hidden_dim, bias = False, gather_output = False, init_method = lambda x : x) 

  def forward(self, x): 
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
  

class TransformerBlock(nn.Module): 
  def __init__(self, layer_id: int, args: ModelArgs):
    super(TransformerBlock, self).__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.attention = Attention(args)
    self.feed_forward = FeedForward(
      dim = args.dim,
      hidden_dim = 4 * args.dim,
      multiple_of = args.multiple_of,
      ffn_dim_multiplier = args.ffn_dim_multiplier,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)

  def forward(
    self, 
    x : torch.Tensor, 
    start_position: int, 
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
  ):
    h = x + self.attention(self.attention_norm(x), start_position, freqs_cis, mask)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out

class Transformer(nn.Module): 
  def __init__(self, params: ModelArgs):
    super(Transformer, self).__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    self.tok_embeddings = VocabParallelEmbedding(
      params.vocab_size, 
      params.dim, 
      init_method = lambda x: x
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(layer_id, params))

    self.norm = RMSNorm(params.dim, eps = params.norm_eps)
    self.output = ColumnParallelLinear(
      params.dim, 
      params.vocab_size, 
      bias = False, 
      init_method = lambda x: x
    )

    self.freqs_cis = precompute_freqs_cis(
      params.dim // params.n_heads,
      params.max_seq_len * 2,
      params.rope_theta,
    )

  @torch.inference_mode() 
  def forward(self, tokens : torch.Tensor, start_position : int): 
    batch_size, seq_len = tokens.shape 
    h = self.tok_embeddings(tokens) 
    self.freqs_cis = self.freqs_cis.to(h.device) 
    freqs_cis = self.freqs_cis[start_position : start_position + seq_len]

    mask = None 
    if seq_len > 1 : 
      mask = torch.full((seq_len, seq_len), float("-inf"), device = tokens.device) 
      mask = torch.triu(mask, diagonal = 1) 
      mask = torch.hstack([torch.zeros((seq_len, start_position), device = tokens.device), mask]).type_as(h)

    for layer in self.layers: 
      h = layer(h, start_position, freqs_cis, mask) 
    h = self.norm(h) 
    out = self.output(h).float() 
    return out 
  

            
        


