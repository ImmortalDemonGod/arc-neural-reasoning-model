# TODO List

- Integrate dropout into the GPT2ARC model architecture:
  - Modify the `FeedForward` class to include `nn.Dropout` layers.
  - Pass the `dropout` parameter from `ModelConfig` to the `FeedForward` and `TransformerBlock` classes.
  - Ensure dropout is applied in appropriate places within the model to prevent overfitting.
