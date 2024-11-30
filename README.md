# ARC Neural Reasoning Model

![CI Tests](https://github.com/ImmortalDemonGod/arc-neural-reasoning-model/actions/workflows/test.yml/badge.svg)

This project implements a neural reasoning model based on the GPT-2 architecture to solve tasks from the Abstraction and Reasoning Corpus (ARC) challenge.

## Installation

### Prerequisites

1. Install UV package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/arc-neural-reasoning-model.git
cd arc-neural-reasoning-model
```

### Installation Options

1. Basic installation (core features only):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. Development installation (includes testing tools):
```bash
uv pip install -e ".[dev]"
```

3. Full installation (all features):
```bash
uv pip install -e ".[dev,train,jupyter]"
```

## Development

### Code Quality

We use several tools to maintain code quality:

1. Format code with Black:
```bash
uv run black gpt2_arc
```

2. Lint with Ruff:
```bash
uv run ruff check gpt2_arc
```

3. Type checking with MyPy:
```bash
uv run mypy gpt2_arc
```

### Testing

Run the tests with coverage:
```bash
uv run pytest --cov=gpt2_arc --cov-report=term-missing
```

### Training

For model training, install training dependencies:
```bash
uv pip install -e ".[train]"
```

Then run training:
```bash
python gpt2_arc/src/training/train_cli.py --max_epochs 100
```

### Jupyter Notebooks

For interactive development:
```bash
uv pip install -e ".[jupyter]"
jupyter lab
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

Please follow our coding standards:
- Use Black for code formatting
- Pass all Ruff linting checks
- Maintain type hints and pass MyPy checks
- Write tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.