#!/usr/bin/env python
"""Gradio web UI for launching LoRA training."""

import argparse
import threading
import time
from typing import List
from pathlib import Path

import gradio as gr
from PIL import Image

from train_lora import train


def run_training(
    train_dir: str,
    output_dir: str,
    pretrained_model: str,
    resolution: int,
    train_batch_size: int,
    learning_rate: float,
    num_epochs: int,
    sample_every: int,
    sample_prompts_text: str,
    lr_scheduler: str,
    rank: int,
    num_inference_steps: int,
):
    prompts = [line.strip() for line in sample_prompts_text.splitlines() if line.strip()]
    args = argparse.Namespace(
        train_dir=train_dir,
        output_dir=output_dir,
        pretrained_model=pretrained_model,
        resolution=int(resolution),
        train_batch_size=int(train_batch_size),
        learning_rate=float(learning_rate),
        num_epochs=int(num_epochs),
        sample_every=int(sample_every),
        sample_prompts=prompts,
        lr_scheduler=lr_scheduler,
        rank=int(rank),
        num_inference_steps=int(num_inference_steps),
    )

    sample_paths: List[str] = []

    def callback(paths: List[str]):
        sample_paths.extend(paths)

    thread = threading.Thread(target=train, args=(args, callback))
    thread.start()
    while thread.is_alive():
        time.sleep(1)
        yield [Image.open(p) for p in sample_paths]
    thread.join()
    yield [Image.open(p) for p in sample_paths]


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Qwen-Image LoRA Trainer")
        with gr.Row():
            train_dir = gr.Textbox(label="Training Directory")
            output_dir = gr.Textbox(label="Output Directory", value="qwen_image_lora")
        pretrained_model = gr.Textbox(label="Pretrained Model", value="Qwen/Qwen-Image")
        with gr.Row():
            resolution = gr.Number(label="Resolution", value=1024)
            train_batch_size = gr.Number(label="Train Batch Size", value=1)
            learning_rate = gr.Number(label="Learning Rate", value=1e-4)
        with gr.Row():
            num_epochs = gr.Number(label="Num Epochs", value=1)
            sample_every = gr.Number(label="Sample Every", value=100)
        sample_prompts = gr.Textbox(
            label="Sample Prompts ('prompt=count' per line)",
            lines=4,
            value="A photo of a dog playing in the park=1",
        )
        with gr.Row():
            lr_scheduler = gr.Dropdown(["constant", "cosine", "linear"], label="LR Scheduler", value="constant")
            rank = gr.Number(label="LoRA Rank", value=4)
            num_inference_steps = gr.Number(label="Validation Steps", value=30)
        start = gr.Button("Start Training")
        gallery = gr.Gallery(label="Intermediate Samples", columns=2)

        start.click(
            run_training,
            inputs=[
                train_dir,
                output_dir,
                pretrained_model,
                resolution,
                train_batch_size,
                learning_rate,
                num_epochs,
                sample_every,
                sample_prompts,
                lr_scheduler,
                rank,
                num_inference_steps,
            ],
            outputs=gallery,
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
