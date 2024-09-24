# TODO List

- Integrate dropout into the GPT2ARC model architecture:
  - Modify the `FeedForward` class to include `nn.Dropout` layers.
  - Pass the `dropout` parameter from `ModelConfig` to the `FeedForward` and `TransformerBlock` classes.
  - Ensure dropout is applied in appropriate places within the model to prevent overfitting.

- Enable hyperparameter optimization from a saved model:
  - Modify `optimize_hyperparameters.py` to load a saved model checkpoint before starting each trial.
  - Add logic to the `objective` function to load the model state from a checkpoint.
  - Ensure the script can accept a model checkpoint path as an argument.
