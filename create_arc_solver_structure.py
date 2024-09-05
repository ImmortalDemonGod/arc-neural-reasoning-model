import os
import sys

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content=""):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_arc_solver_structure(base_path):
    project_structure = {
        "src": {
            "sat_solver": ["cnf_converter.py", "pycosat_wrapper.py", "solver.py"],
            "reinforcement_learning": ["strategies.py", "learner.py"],
            "arc_tasks": ["task_loader.py", "task_solver.py"],
            "utils": ["grid_operations.py"]
        },
        "tests": {
            "sat_solver": ["test_cnf_converter.py", "test_pycosat_wrapper.py", "test_solver.py"],
            "reinforcement_learning": ["test_strategies.py", "test_learner.py"],
            "arc_tasks": ["test_task_loader.py", "test_task_solver.py"],
            "utils": ["test_grid_operations.py"]
        },
        "data": {
            "arc_tasks": ["training", "evaluation"]
        },
        "docs": ["architecture.md", "sat_solver.md", "reinforcement_learning.md"]
    }

    for directory, contents in project_structure.items():
        dir_path = os.path.join(base_path, directory)
        create_directory(dir_path)
        
        if isinstance(contents, list):
            for file in contents:
                create_file(os.path.join(dir_path, file))
        elif isinstance(contents, dict):
            for subdir, files in contents.items():
                subdir_path = os.path.join(dir_path, subdir)
                create_directory(subdir_path)
                for file in files:
                    if '.' in file:  # It's a file
                        create_file(os.path.join(subdir_path, file))
                    else:  # It's a directory
                        create_directory(os.path.join(subdir_path, file))
        
        create_file(os.path.join(dir_path, "__init__.py"))

    # Create root level files
    root_files = ["requirements.txt", "setup.py", "README.md"]
    for file in root_files:
        create_file(os.path.join(base_path, file))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = os.path.join(os.getcwd(), "arc_solver_project")
    
    create_arc_solver_structure(base_path)
    print(f"ARC Solver project structure created at: {base_path}")
