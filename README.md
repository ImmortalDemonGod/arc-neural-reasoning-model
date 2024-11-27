# ARC Neural Reasoning Model

![CI Tests](https://github.com/ImmortalDemonGod/arc-neural-reasoning-model/actions/workflows/test.yml/badge.svg)

This project implements a neural reasoning model based on the GPT-2 architecture to solve tasks from the Abstraction and Reasoning Corpus (ARC) challenge.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/arc-neural-reasoning-model.git
cd arc-neural-reasoning-model
pip install -e .
```

For development, install the extra dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

[Provide basic usage instructions here once the model is implemented]

## Testing

Run the tests using pytest:

```bash
pytest
```

## Contributing

[Add contribution guidelines here]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

aider --cache-prompts --no-stream

DEV RUNNING COMMAND: 
rm -rf /workspaces/arc-neural-reasoning-model/results;rm -rf /workspaces/arc-neural-reasoning-model/checkpoints;rm -rf /workspaces/arc-neural-reasoning-model/cache;rm -rf /workspaces/arc-neural-reasoning-model/runs;pip install -e . ;python3 /workspaces/arc-neural-reasoning-model/gpt2_arc/src/training/train_cli.py --max_epochs 1 --fast_dev_run --model_checkpoint /workspaces/arc-neural-reasoning-model/final_model_ab9f1610-de99-44e7-a844-4a3b6480ad08.pth --log_level debug