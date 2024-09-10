def differential_pixel_accuracy(input, target, prediction):
    # Find pixels that differ between input and target
    input_target_diff = input != target

    # Count how many of these differing pixels the model predicted correctly
    correct_diff_predictions = (prediction != input) & input_target_diff

    # Calculate accuracy
    total_diff_pixels = input_target_diff.sum().item()
    correct_diff_pixels = correct_diff_predictions.sum().item()

    if total_diff_pixels > 0:
        accuracy = correct_diff_pixels / total_diff_pixels
    else:
        accuracy = 1.0  # If no pixels differ, consider it 100% accurate

    return accuracy, input_target_diff, correct_diff_predictions
