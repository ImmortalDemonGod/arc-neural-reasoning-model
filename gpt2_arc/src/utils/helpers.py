# gpt2_arc/src/utils/helpers.py
def differential_pixel_accuracy(input, target, prediction):
    print(f"Differential pixel accuracy - Input shape: {input.shape}, Target shape: {target.shape}, Prediction shape: {prediction.shape}")
    
    # Ensure all inputs have the same shape
    assert input.shape == target.shape == prediction.shape, f"Shape mismatch: input {input.shape}, target {target.shape}, prediction {prediction.shape}"
    
    input = input.view(-1)
    target = target.view(-1)
    prediction = prediction.view(-1)

    input_target_diff = input != target
    correct_diff_predictions = (prediction != input) & input_target_diff

    total_diff_pixels = input_target_diff.sum().item()
    correct_diff_pixels = correct_diff_predictions.sum().item()

    print(f"Total different pixels: {total_diff_pixels}")
    print(f"Correctly predicted different pixels: {correct_diff_pixels}")

    if total_diff_pixels > 0:
        accuracy = correct_diff_pixels / total_diff_pixels
    else:
        accuracy = 1.0

    print(f"Calculated accuracy: {accuracy}")
    return accuracy, input_target_diff, correct_diff_predictions
