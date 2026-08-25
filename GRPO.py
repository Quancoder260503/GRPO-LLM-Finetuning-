import numpy as np 
import torch 
import dataclasses
from CountdownDataset import * 
from Qwen import Transformer as qwen_transformer 
from Llama import Transformer as llama_transformer
from typing import Optional
from collections import defaultdict
import gc
import math 

@torch.no_grad() 
def collecting_trajectories(
  curr_policy, 
  tokenizer,
  max_gen_len : int, 
  num_answer_per_question : int, 
  reward_function : callable, 
  device : torch.device, 
  dtype : torch.dtype,  
  batch : MiniBatch, 
) -> List[Episode]:
  end_token = tokenizer.eos_token 
  end_token_id = tokenizer.eos_token_id 
  pad_token_id = tokenizer.pad_token_id
  prefix_token_ids = batch.prefix_token_ids

  batch_size = len(batch.prefix) * num_answer_per_question
  min_prompt_len = min(len(t) for t in prefix_token_ids)
  max_prompt_len = max(len(t) for t in prefix_token_ids)
  total_len = max_gen_len + max_prompt_len

  curr_policy.init_kv_cache(
   max_batch_size = batch_size, 
   max_seq_len = total_len, 
   device = device, 
   dtype = dtype, 
  )

  tokens = torch.full((batch_size, total_len), pad_token_id, dtype = torch.long, device = device)

  for k, t in enumerate(prefix_token_ids): 
    offset = k * num_answer_per_question 
    for i in range(num_answer_per_question):
      tokens[offset + i, : len(t)] = torch.tensor(t, dtype = torch.long, device = device) 

  prev_pos = 0
  input_text_mask = tokens != pad_token_id 

  is_finished = torch.zeros((batch_size, ), dtype = torch.bool, device = device) 

  for curr_pos in range(min_prompt_len, total_len):
    print(
      f"\r* Generating trajectories: {curr_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
      flush = True,
      end = "",
    )

    with torch.autocast(device_type = device.type, dtype = dtype):
      logits = curr_policy.inference(tokens[:, prev_pos : curr_pos], prev_pos)

    probs = torch.softmax(logits[:, -1], dim = -1)
    next_token = torch.multinomial(probs, num_samples = 1)
    next_token = next_token.reshape(-1)
    next_token = torch.where(input_text_mask[:, curr_pos], tokens[:, curr_pos], next_token)

    tokens[:, curr_pos] = next_token
    if end_token_id is not None: 
      is_end_token = next_token == end_token_id 
      is_generated_token = ~input_text_mask[:, curr_pos]
      is_finished |= (is_end_token & is_generated_token)
    prev_pos = curr_pos 
    if is_finished.all():
      break 

  curr_policy.del_kv_cache()
  gc.collect()
  torch.cuda.empty_cache()
  is_finished_list = is_finished.tolist()
  tokens_list = tokens.tolist()

  episodes = [] 

  for i in range(batch_size // num_answer_per_question): 
    for j in range(num_answer_per_question):
      idx = i * num_answer_per_question + j 
      generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) : ]

      if pad_token_id in generated_token_ids:
        generated_token_ids = generated_token_ids[: generated_token_ids.index(pad_token_id)]

      generated_text = tokenizer.detokenize(generated_token_ids)
      rewards = reward_function(
        response = generated_text, 
        number = batch.numbers[i], 
        target = batch.target[i], 
        end_token = end_token
      )
      episode = Episode(
        prefix = batch.prefix[i],
        text = batch.prefix[i] + generated_text,
        prefix_token_ids = batch.prefix_token_ids[i],
        prefix_tokens = batch.prefix_tokens[i],
        generated_token_ids = generated_token_ids,
        is_finished = is_finished_list[idx],
        reward = rewards["reward"],
        reward_info = rewards["reward_info"],
      )
      episodes.append(episode)

  print("\r", end=" " * 100, flush = True)
  return episodes


def normalize_rewards_per_group(episodes : List[Episode]) -> List[Episode]:
  groups = defaultdict(list)
  for episode in episodes: 
    groups[tuple(episode.prefix)].append(episode)

  output = []
  for group in groups.values():
    group_rewards = [item.reward for item in group]
    mean_reward = np.mean(group_rewards)
    std_reward = np.std(group_rewards)

    for episode in group: 
      normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4) 
      episode = dataclasses.replace(episode, reward = normalized_reward)
      output.append(episode)
  return output


def compute_entropy(logits : torch.Tensor) -> torch.Tensor: 
  probs = torch.nn.functional.softmax(logits, dim = -1)
  entropy = torch.logsumexp(logits, dim = -1) - torch.sum(probs * logits, dim = -1)
  return entropy

def update_policy(
  curr_policy,
  prev_policy,
  reference_policy, 
  optimizer, 
  episodes : List[Episode], 
  micro_batch_size : int, 
  pad_token_id : int, 
  max_grad_norm : float, 
  device : torch.device, 
  dtype : torch.dtype, 
  grpo_eps_clip : float, 
  kl_weight : float, 
  entropy_weight : float
):
  episodes = normalize_rewards_per_group(episodes)
  episodes.sort(key = lambda x : len(x.prefix_token_ids) + len(x.generated_token_ids))
  num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)

  total_entropy = 0.0 
  total_obj = 0.0 
  total_kl = 0.0 
  num_itr = 0

  for i in range(0, len(episodes), micro_batch_size): 
    print(
     f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}", flush = True, end = "",
    )
    j = min(i + micro_batch_size, len(episodes))
    batch_episodes = episodes[i:j]
    batch_lengths = [len(episode.prefix_token_ids) + len(episode.generated_token_ids) for episode in batch_episodes]
    batch_max_length = max(batch_lengths)

    batch_token_ids = [
      episode.prefix_token_ids + 
      episode.generated_token_ids + 
      [pad_token_id] * (batch_max_length - batch_lengths[i]) 
      for i, episode in enumerate(batch_episodes)
    ]

    batch_masks = [
      [0] * len(episode.prefix_token_ids) + 
      [1] * len(episode.generated_token_ids) + 
      [0] * (batch_max_length - batch_lengths[i]) 
      for i, episode in enumerate(batch_episodes)
    ]

    
    batch_advantages = [episode.reward for episode in batch_episodes]
    batch_token_ids = torch.tensor(batch_token_ids, device = device, dtype = torch.long) 
    batch_masks = torch.tensor(batch_masks, device = device, dtype = torch.bool)

    batch_advantages = torch.tensor(batch_advantages, device = device, dtype = torch.float32)
    with torch.autocast(device_type = device.type, dtype = dtype):
      input_token_ids = batch_token_ids[:, :-1]
      target_token_ids = batch_token_ids[:, 1:]
      target_masks = batch_masks[:, 1:]


    # old_policy 
    with torch.no_grad():
      logits_old = prev_policy.forward(input_token_ids).float()
      log_prob_old = -torch.nn.functional.cross_entropy(
        logits_old.reshape(-1, logits.size(-1)), 
        target_token_ids.reshape(-1), 
        ignore_index = pad_token_id, 
        reduction = "none"
      ).reshape(input_token_ids.shape[0], -1).detach()

   
    logits = curr_policy.forward(input_token_ids).float()
    log_probs = -torch.nn.functional.cross_entropy(
      logits.reshape(-1, logits.size(-1)), 
      target_token_ids.reshape(-1), 
      ignore_index = pad_token_id, 
      reduction = "none"
    ).reshape(input_token_ids.shape[0], -1)
    

    ratio = torch.exp(log_probs - log_prob_old) 
    surr1 = ratio * batch_advantages[:, None]
    surr2 = torch.clamp(ratio, 1 - grpo_eps_clip, 1 + grpo_eps_clip) * batch_advantages[:, None]

    clipped_surrogate_objective = torch.min(surr1, surr2)
    clipped_surrogate_objective = (clipped_surrogate_objective * target_masks).sum() / num_target_tokens

    # reference_policy 
    with torch.no_grad():
      logits_ref = reference_policy.forward(input_token_ids).float()
      log_probs_ref = -torch.nn.functional.cross_entropy(
        logits_ref.reshape(-1, logits.size(-1)), 
        target_token_ids.reshape(-1), 
        ignore_index = pad_token_id, 
        reduction = "none"
      ).reshape(input_token_ids.shape[0], -1).detach()

    log_ratio_ref = log_probs - log_probs_ref
    kl_div_estimate = torch.exp(log_ratio_ref) - log_ratio_ref - 1. 
    kl_div_estimate = (kl_div_estimate * target_masks).sum() / num_target_tokens

   
    token_entropy = compute_entropy(logits)
    token_entropy = (token_entropy * target_masks).sum() / num_target_tokens

    loss = -clipped_surrogate_objective - token_entropy * entropy_weight + kl_div_estimate * kl_weight

    loss.backward()

    total_kl += kl_div_estimate.item()
    total_obj += clipped_surrogate_objective.item() 
    total_entropy += token_entropy.item()
    num_itr += 1 

  grad_norm = torch.nn.utils.clip_grad_norm_(curr_policy.parameters(), max_norm = max_grad_norm)
  optimizer.step()
  optimizer.zero_grad(set_to_none = True)
  return {
    "surr_obj": total_obj / num_itr,
    "grad_norm": grad_norm.item(), 
    "entropy": total_entropy / num_itr, 
    "kl" : total_kl / num_itr, 
  }

