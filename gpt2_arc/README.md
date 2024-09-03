# GPT-2 ARC Neural Reasoning Model

This project implements a neural reasoning model based on the GPT-2 architecture to solve tasks from the Abstraction and Reasoning Corpus (ARC) challenge.

## Features

- **Data Handling**: Utilizes a custom `ArcDataset` class for handling and preprocessing ARC data.
- **Model Architecture**: Implements a `GPT2ARC` model leveraging the pre-trained GPT-2 architecture.
- **Training**: Includes a `train.py` script for training the model using PyTorch Lightning, with support for logging and checkpointing.
- **Testing**: Comprehensive test suite using `pytest` to ensure model and data integrity.

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

### Training the Model

To train the model, use the following command:

```
python src/train.py --train_data path/to/train_data --val_data path/to/val_data --batch_size 32 --learning_rate 1e-4 --max_epochs 10 --use_gpu
```

Adjust the parameters as needed. The trained model checkpoints will be saved in the `checkpoints` directory.

### Evaluating the Model

To evaluate a trained model on a test set, use the following command:

```
python src/evaluate.py --test_data path/to/test_data --model_checkpoint path/to/model_checkpoint.ckpt --batch_size 32
```

This will output the evaluation metrics for the model on the test dataset.

## Running Tests

To run the tests, use the following command:

```
pytest -v
```

This will run all tests and display the results, including test coverage.

## Contributing

[Add contribution guidelines here]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
