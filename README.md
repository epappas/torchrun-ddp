# torchrun-ddp

PyTorch Distributed Data Parallel (DDP) training framework for fine-tuning LLaMA models with elastic training support and gradient compression.

## Features

- **Distributed Data Parallel Training**: Scale training across multiple GPUs and nodes using PyTorch DDP
- **Elastic Training**: Fault-tolerant training with automatic worker recovery via torchrun
- **DCT Gradient Compression**: Reduces gradient communication bandwidth by ~50% using Discrete Cosine Transform
- **LLaMA Fine-tuning**: Pre-configured for LLaMA-2-7B with HuggingFace Transformers
- **Streaming Dataset**: Memory-efficient training with streamed C4 dataset and automatic sharding
- **Advanced LR Scheduling**: Linear warmup with cosine annealing restarts

## Requirements

- Python >= 3.11
- CUDA-capable GPU(s)
- NVIDIA drivers with CUDA support
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- HuggingFace account with access to LLaMA models

## Installation

```bash
# Clone the repository
git clone https://github.com/epappas/torchrun-ddp.git
cd torchrun-ddp

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token_here
```

You need a HuggingFace token with access to the LLaMA model. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

### Single Node, Single GPU

```bash
torchrun --nproc_per_node=1 train.py
```

### Single Node, Multiple GPUs

```bash
torchrun --nproc_per_node=4 train.py
```

### Multi-Node Training

On the master node:

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=training_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master_ip:29400 \
    train.py
```

On worker nodes:

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=training_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master_ip:29400 \
    train.py
```

### Using run.sh

Alternative wrapper script with configurable environment variables:

```bash
# Single node, single GPU
./run.sh

# Single node, 4 GPUs
NPROC_PER_NODE=4 ./run.sh

# Multi-node (master node)
NNODES=2 NPROC_PER_NODE=4 NODE_RANK=0 MASTER_ADDR=master_ip ./run.sh

# Multi-node (worker node)
NNODES=2 NPROC_PER_NODE=4 NODE_RANK=1 MASTER_ADDR=master_ip ./run.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NPROC_PER_NODE` | `1` | GPUs per node |
| `NNODES` | `1` | Total number of nodes |
| `NODE_RANK` | `0` | Rank of this node |
| `MASTER_ADDR` | `localhost` | Master node address |
| `MASTER_PORT` | `29400` | Master node port |
| `RDZV_ID` | `default` | Rendezvous ID for the job |

### Environment Variables

These are automatically set by `torchrun`:

| Variable | Description |
|----------|-------------|
| `RANK` | Global rank of the process |
| `WORLD_SIZE` | Total number of processes |
| `LOCAL_RANK` | Rank within the local node |
| `LOCAL_WORLD_SIZE` | Number of processes on local node |
| `MASTER_ADDR` | Address of the master node |
| `MASTER_PORT` | Port for distributed communication |

## Architecture

```
torchrun-ddp/
  train.py                     # Entry point - reads torchrun env vars
  run.sh                       # Wrapper script for torchrun
  pyproject.toml               # Project configuration and dependencies
  src/torchrun_ddp/
      __init__.py              # Package exports
      main.py                  # Core DDP training loop
      model.py                 # Model definitions
      optimizer.py             # AdamW optimizer with LR scheduling
      training_data.py         # Dataset loading and sharding
      compression.py           # DCT gradient compression hook
```

### Components

**main.py** - Core training orchestration:

- Initializes distributed process group (gloo backend)
- Loads and wraps model with DDP
- Registers gradient compression hooks
- Manages training loop with gradient accumulation
- Handles checkpointing (rank 0 only)

**compression.py** - Gradient compression:

- Implements DCT-based compression hook
- Transforms gradients to frequency domain
- Discards high-frequency components (50% reduction)
- Decompresses after all-reduce

**optimizer.py** - Optimizer configuration:

- AdamW with weight decay (0.01)
- Linear warmup (250 steps)
- Cosine annealing with restarts

**training_data.py** - Data pipeline:

- Streams from allenai/c4 dataset
- Automatic sharding via `split_dataset_by_node`
- Configurable sample size

## Training Parameters

Modify parameters in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | `meta-llama/Llama-2-7b-hf` | HuggingFace model identifier |
| `learning_rate` | `1e-5` | Base learning rate |
| `seq_length` | `512` | Maximum sequence length |
| `batch_size` | `4` | Per-GPU batch size |
| `accumulation_steps` | `1` | Gradient accumulation steps |
| `epochs` | `1` | Number of training epochs |

## Development

### Running Tests

```bash
uv run pytest
uv run pytest -k test_name  # Run specific test
```

### Linting and Formatting

```bash
uv run ruff check .   # Lint
uv run ruff format .  # Format
```

## Output

Training outputs are saved to:

- Checkpoints: `$TMPDIR/model.checkpoint`
- Final model: `./final_model/`

Logs include per-step metrics:

- Loss
- Learning rate
- Throughput (tokens/s)
- GPU memory usage

## License

MIT License

## Author

Evangelos Pappas (<epappas@evalonlabs.com>)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and linting (`uv run pytest && uv run ruff check .`)
5. Commit your changes (`git commit -m 'Add your feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request
