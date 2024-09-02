import os

def create_project_structure(base_path='gpt2_arc'):
    # Define the directory structure
    structure = {
        'src': {
            'data': ['__init__.py', 'arc_dataset.py'],
            'models': ['__init__.py', 'gpt2.py'],
            'training': ['__init__.py', 'trainer.py'],
            'utils': ['__init__.py', 'helpers.py'],
        },
        'tests': ['__init__.py', 'test_arc_dataset.py', 'test_gpt2.py', 'test_trainer.py'],
        '': ['requirements.txt', 'setup.py', 'README.md'],
    }

    # Create the directories and files
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        if isinstance(files, list):  # If files is a list, it's a direct file in a folder
            for file in files:
                with open(os.path.join(folder_path, file), 'w') as f:
                    pass
        elif isinstance(files, dict):  # If files is a dict, it represents subdirectories
            for subfolder, subfiles in files.items():
                subfolder_path = os.path.join(folder_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                for subfile in subfiles:
                    with open(os.path.join(subfolder_path, subfile), 'w') as f:
                        pass

    print(f"Project structure created at: {os.path.abspath(base_path)}")

# Run the function to create the structure
create_project_structure()