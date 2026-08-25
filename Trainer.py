import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn 
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from CountdownDataset import CountdownTaskDataset, reward_function
from GRPO import update_policy, collecting_trajectories
from Qwen import Transformer as qwen_transformer
from QwenTokenizer import Tokenizer as qwen_tokenizer
import tqdm 
import copy 


def evaluate(model, tokenizer, device, dtype, config):
  test_dataset = CountdownTaskDataset(
    data_path=config["data"]["path"],
    tokenizer = tokenizer,
    split = "test",
    test_size = config["data"]["test_size"],
  )
  generator = torch.Generator(device = device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
  dataloader = DataLoader(
    test_dataset,
    shuffle = False,
    collate_fn = CountdownTaskDataset.collate_fn,
    generator = generator,
    batch_size = config["training"]["batch_size"] // 2,
    drop_last = False,
  )
  success = []
  for batch in dataloader:
    episodes = collecting_trajectories(
      model = model,
      tokenizer = tokenizer,
      batch = batch,
      max_gen_len = config["training"]["max_gen_len"] * 2,
      num_answer_per_question = 1,
      reward_function = reward_function,
      device = device,
      dtype = dtype,
    )
    success.extend([episode.reward_info["answer_reward"] for episode in episodes])
  return np.mean(success)


def main(config_path: str):
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)

  pretrained_model_path = Path(config["model"]["pretrained_model_path"])
  device = torch.device(config["model"]["device"])
  if device.type == "cuda":
    try:
      torch.empty(1, device = device)
    except RuntimeError:
      print("CUDA is unavailable in this PyTorch installation; falling back to CPU.")
      device = torch.device("cpu")
  dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
  }
  dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
  if device.type == "cpu":
    dtype = torch.float32
  torch.set_default_device(device)
  torch.random.manual_seed(config["training"]["random_seed"])

  BATCH_SIZE = config["training"]["batch_size"]
  NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
  NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

  current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
  tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
  tokenizer = qwen_tokenizer(str(pretrained_model_path / "tokenizer.json"))

  train_dataset = CountdownTaskDataset(
    data_path = config["data"]["path"],
    tokenizer = tokenizer,
    split = "train",
    test_size = config["data"]["test_size"],
  )
  generator = torch.Generator(device = device)
  train_dataloader = DataLoader(
    train_dataset,
    shuffle = True,
    collate_fn = CountdownTaskDataset.collate_fn,
    generator = generator,
    batch_size=NUM_QUESTIONS_PER_BATCH,
  )

  model = qwen_transformer.from_pretrained(pretrained_model_path, device = device).train()

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = config["training"]["learning_rate"],
    weight_decay = config["training"]["weight_decay"],
    betas = config["training"]["betas"],
   )

  start_time = time.time()
  ckpt_dir = Path(config["training"]["ckpt_dir"])
  ckpt_dir.mkdir(parents = True, exist_ok = True)

  reference_model = copy.deepcopy(model)

  for param in reference_model.parameters(): 
    param.requires_grad_(False)
  reference_model.eval()

  prev_model = copy.deepcopy(model)
  for step, batch in tqdm.tqdm(enumerate(train_dataloader, start = 1)):
    prev_model.eval()
    episodes = collecting_trajectories(
      curr_policy = prev_model,
      tokenizer = tokenizer,
      batch = batch,
      max_gen_len = config["training"]["max_gen_len"],
      num_answer_per_question = NUM_ANSWERS_PER_QUESTION,
      reward_function = reward_function,
      device = device,
      dtype = dtype,
    )
    if config["training"]["skip_unfinished_episodes"]:
      episodes = [episode for episode in episodes if episode.is_finished]

    model.train()
    results = update_policy(
      curr_policy = model,
      reference_policy = reference_model, 
      optimizer = optimizer,
      episodes = episodes,
      micro_batch_size = config["training"]["micro_batch_size"],
      pad_token_id = tokenizer.pad_token_id,
      max_grad_norm = config["training"]["max_grad_norm"],
      device = device,
      dtype = dtype,
      grpo_eps_clip = config['training']['grpo_eps_clip'], 
      kl_weight = config['training']['kl_weight'], 
      entropy_weight = config['training']['entropy_weight']
    )

    prev_model.load_state_dict(model.state_dict())

    torch.cuda.synchronize()
    end_time = time.time()
    duration = end_time - start_time
    start_time = end_time

    # compute and log important metrics
    reward = [episode.reward for episode in episodes]
    formatted_reward = [episode.reward_info["format_reward"] for episode in episode]
    answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
    num_finished_episodes = sum(episode.is_finished for episode in episodes)
    mean_reward = np.mean(reward)
    std_reward = np.std(reward)
    success_rate = np.mean(answer_reward)
    format_reward = np.mean(formatted_reward)
    grad_norm = results["grad_norm"]
    entropy = results["entropy"]
    kl_div = results['kl']
    surr_obj = results['surr_obj']
    lr = optimizer.param_groups[0]["lr"]
    loss = results["loss"]
    mean_response_len = np.mean([len(episode.generated_token_ids) for episode in episodes])
    print(
      f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
      f"train success_rate: {success_rate:.2f}, "
      f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
      f"num_finished_episodes: {num_finished_episodes}, "
      f"mean_response_len: {mean_response_len:.2f}, "
      f"clipped surrogate objective: {surr_obj:.4f}"
      f"kl divergence : {kl_div:.4f}"
      f"entropy: {entropy:.2f}"
    )
    if step % config["training"]["eval_interval"] == 0:
      model.eval()
      eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
      print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
      tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

    tb_writer.add_scalar("loss", loss, step)
    tb_writer.add_scalar("mean_reward", mean_reward, step)
    tb_writer.add_scalar("std_reward", std_reward, step)
    tb_writer.add_scalar("success_rate/train", success_rate, step)
    tb_writer.add_scalar("format_reward", format_reward, step)
    tb_writer.add_scalar("grad_norm", grad_norm, step)
    tb_writer.add_scalar("duration", duration, step)
    tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
    tb_writer.add_scalar("learning_rate", lr, step)
    tb_writer.add_scalar("kl_div", kl_div, step) 
    tb_writer.add_scalar("clipped_surrogate_objective", surr_obj, step) 
    tb_writer.add_scalar("mean_response_len", mean_response_len, step)
    tb_writer.add_scalar("entropy", entropy, step)

    for i, episode in enumerate(episodes):
        # TensorBoard treats text as markdown.
      text = html.escape(episode.text)
      tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

    # save checkpoint
    if step % config["training"]["ckpt_save_interval"] == 0:
      output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
      torch.save(model.state_dict(), output_file)
      print(f"Saved checkpoint to {output_file}")

    


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--config", type = str, default = "Config/QwenConfig.yaml")
  args = parser.parse_args()
  main(args.config)