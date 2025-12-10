# AGENTS.md

Instructions for AI coding agents working on this PyTorch DDP training project.

## Dev environment setup

- Use `uv sync` to install all dependencies into the local `.venv`.
- Python version is pinned to 3.13 in `.python-version`.
- Create a `.env` file with `HF_TOKEN=<your_token>` for HuggingFace authentication.
- The package is installed in editable mode; changes to `src/torchrun_ddp/` are immediately available.

## Code style

- Use `uv run ruff check .` to lint and `uv run ruff format .` to format.
- Max cyclomatic complexity is 10 (configured in `pyproject.toml`).
- All functions must be typed. No `Any` types unless absolutely necessary.
- Prefer early returns over nested conditionals.
- Keep functions under 50 lines.
- No emojis in code or comments.

## Testing instructions

- Run `uv run pytest` to execute the full test suite.
- Run `uv run pytest -k "<test_name>"` to run a specific test.
- Tests require GPU access for full integration testing.
- Never mock or stub core PyTorch/DDP functionality in tests.

## Running the training

- Single GPU: `torchrun --nproc_per_node=1 train.py`
- Multi-GPU: `torchrun --nproc_per_node=<N> train.py`
- Use `./run.sh` wrapper with env vars: `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`.
- Training requires CUDA-capable GPU and ~14GB VRAM for LLaMA-2-7B.

## Project structure

- `train.py` - Entry point, reads torchrun environment variables.
- `src/torchrun_ddp/main.py` - Core training loop with DDP setup.
- `src/torchrun_ddp/compression.py` - DCT gradient compression hook.
- `src/torchrun_ddp/optimizer.py` - AdamW with warmup + cosine scheduler.
- `src/torchrun_ddp/training_data.py` - C4 dataset loading with node sharding.

## Key implementation details

- Process group uses `gloo` backend (CPU-based, works without NCCL).
- Gradient compression reduces communication by ~50% via DCT transform.
- Dataset is automatically sharded across nodes using `split_dataset_by_node`.
- Checkpoints are saved only on rank 0 to avoid conflicts.
- Final model saved to `./final_model/` directory.

## Commit guidelines

- Format: `{type}({component}): {description}`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Run linting and tests before committing.
- Never include AI assistant names in commit messages.
