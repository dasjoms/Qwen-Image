#!/usr/bin/env python
"""LoRA fine-tuning script for Qwen-Image.

This script expects a directory of images with matching text files::

    dataset/
      001.png
      001.txt
      002.png
      002.txt

Each ``*.txt`` file should contain the caption for the image with the same base
name. During training the script periodically saves sample images so that
intermediate results can be inspected.
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model


class ImageTextDataset(Dataset):
    """Simple dataset that pairs images with text captions."""

    def __init__(self, folder: str, resolution: int = 1024) -> None:
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"{folder} does not exist")
        self.images: List[Path] = sorted(
            [p for p in self.folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=Image.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            raise FileNotFoundError(f"Missing caption file {caption_path}")
        caption = caption_path.read_text(encoding="utf-8").strip()
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return {"pixel_values": image, "caption": caption}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen-Image")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with image/txt pairs")
    parser.add_argument("--output_dir", type=str, default="qwen_image_lora", help="Where to store outputs")
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image", help="Base model repo or path")
    parser.add_argument("--resolution", type=int, default=1024, help="Training resolution")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--sample_every",
        type=int,
        default=100,
        help="Save a sample image every N steps",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="A photo of a dog playing in the park",
        help="Prompt used for intermediate sampling",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "cosine", "linear"],
        help="Learning rate scheduler",
    )
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Steps used when generating validation images",
    )
    return parser.parse_args()


def save_sample(pipeline: DiffusionPipeline, prompt: str, step: int, out_dir: str) -> None:
    """Generate and save an image using the current model."""
    pipeline.unet.eval()
    image = pipeline(prompt=prompt, num_inference_steps=30).images[0]
    save_path = os.path.join(out_dir, f"sample_{step:05d}.png")
    image.save(save_path)
    pipeline.unet.train()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ImageTextDataset(args.train_dir, resolution=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    pipeline.to(accelerator.device)

    # Freeze everything except the added LoRA layers
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=args.learning_rate)
    num_training_steps = len(dataloader) * args.num_epochs
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    tokenizer = pipeline.tokenizer
    global_step = 0

    for epoch in range(args.num_epochs):
        for batch in dataloader:
            captions = batch["caption"]
            pixel_values = batch["pixel_values"].to(accelerator.device)

            with accelerator.accumulate(pipeline.unet):
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device, dtype=torch.long
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                tokens = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                encoder_hidden_states = pipeline.text_encoder(tokens)[0]

                model_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % args.sample_every == 0:
                save_sample(pipeline, args.validation_prompt, global_step, args.output_dir)

            global_step += 1

    if accelerator.is_main_process:
        lora_path = os.path.join(args.output_dir, "lora")
        pipeline.unet.save_pretrained(lora_path)


if __name__ == "__main__":
    main()
