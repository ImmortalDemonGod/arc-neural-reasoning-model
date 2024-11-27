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
pip install -r  gpt2_arc/requirements.txt
```

## Usage

[Provide basic usage instructions here once the model is implemented]

## Testing

Run the tests using pytest:

```bash
pytest -v
```

## Contributing

## Contributing

We welcome contributions! Follow these guidelines to help us improve the project:

1. **Fork the Repository**  
   Click the "Fork" button at the top of this repository to create your own copy.

2. **Clone Your Fork**  
   ```bash
   git clone https://github.com/your-username/arc-neural-reasoning-model.git
   cd arc-neural-reasoning-model
   ```

3. **Create a New Branch**  
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes Using `aider`**  
   To simplify modifications, use `aider` to describe your intended changes in natural language:
   ```bash
   aider --weak-model gpt-4o-mini --auto-lint --model o1-mini --architect --editor-model gpt-4o --pretty
   ```
   Follow the prompts to specify your changes, and `aider` will assist in implementing them.

5. **Commit Your Changes**  
   ```bash
   git add .
   git commit -m "Describe your changes here"
   ```

6. **Push to Your Fork**  
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**  
   Go to the original repository and click "Compare & pull request". Provide a clear description of your changes and the problem they address.

**Guidelines:**
- Follow the [PEP 8](https://pep8.org/) coding style.
- Write clear and concise commit messages.
- Include documentation updates if your changes affect usage or functionality.
- Respect the project's code of conduct.

Thank you for contributing to the ARC Neural Reasoning Model!


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

aider --cache-prompts --no-stream

DEV RUNNING COMMAND: 
rm -rf /workspaces/arc-neural-reasoning-model/results;rm -rf /workspaces/arc-neural-reasoning-model/checkpoints;rm -rf /workspaces/arc-neural-reasoning-model/cache;rm -rf /workspaces/arc-neural-reasoning-model/runs;pip install -e . ;python3 /workspaces/arc-neural-reasoning-model/gpt2_arc/src/training/train_cli.py --max_epochs 1 --fast_dev_run --model_checkpoint /workspaces/arc-neural-reasoning-model/final_model_ab9f1610-de99-44e7-a844-4a3b6480ad08.pth --log_level debug
