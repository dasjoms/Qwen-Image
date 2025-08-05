# Guidance for future contributors

- `src/train_lora.py` provides a minimal LoRA fine-tuning script for Qwen-Image.
- Training data should live in a folder where each image file (e.g. `0001.png`) has a
  text file with the same stem (e.g. `0001.txt`) containing its prompt or caption.
- Use `accelerate launch` to run distributed training if needed:
  ```bash
  accelerate launch src/train_lora.py --train_dir /path/to/data --output_dir output
  ```
- Intermediate sample images are saved in the output directory every `--sample_every` steps.
- To check syntax run:
  ```bash
  python -m py_compile src/train_lora.py
  ```
