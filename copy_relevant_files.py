import os
import subprocess

def copy_to_clipboard(text):
    """Copy the given text to the clipboard."""
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(text.encode('utf-8'))

relevant_files_mapping = {
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_benchmark.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/helpers.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/benchmark.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_gpt2.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_end_to_end.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/evaluate.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/benchmark.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/train.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_arc_dataset.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_differential_pixel_accuracy.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/helpers.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_train.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/train.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/benchmark.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/evaluate.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_trainer.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_results_collector.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_model_evaluation.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/helpers.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py"
    ],
    "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/test_integration_experiment.py": [
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
        "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py"
    ]
}

def main():
    test_files = list(relevant_files_mapping.keys())

    print("Select a test file to copy relevant files to clipboard:")
    for idx, test_file in enumerate(test_files, start=1):
        print(f"{idx}. {test_file}")

    choice = int(input("Enter the number of the test file: ")) - 1
    if choice < 0 or choice >= len(test_files):
        print("Invalid choice. Exiting.")
        return

    selected_test_file = test_files[choice]
    relevant_files = relevant_files_mapping[selected_test_file]

    combined_content = ""
    for file_path in relevant_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                combined_content += f"\n\n# {file_path}\n\n"
                combined_content += f.read()
        else:
            print(f"File not found: {file_path}")

    copy_to_clipboard(combined_content)
    print("Relevant files' contents copied to clipboard.")

if __name__ == "__main__":
    main()
