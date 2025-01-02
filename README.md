# EasyTrain

A simplified script for training LLaMA models.

## Quick Start

1. Make the launch script executable:

```bash
chmod +x launch.sh
```

2. Run the script:

```bash
./launch.sh
```



## Requirements

Before running, ensure you have:
- Python 3.8+
- PyTorch 2.0+
- Transformers library
- DeepSpeed

## Environment Variables

Set these before running:
- `HF_TOKEN`: Your Hugging Face token
- `WANDB_API_KEY`: Your Weights & Biases API key (for logging)

## Features

- Distributed training with DeepSpeed
- ZeRO-3 optimization
- Automatic mixed precision (AMP)
- Gradient checkpointing