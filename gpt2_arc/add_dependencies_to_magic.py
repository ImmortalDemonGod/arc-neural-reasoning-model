import subprocess
import re
import sys
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    filename='magic_add_dependencies.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def parse_requirements(file_path):
    """
    Parses a requirements.txt file and returns a list of (package, version_spec) tuples.
    """
    requirements = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Remove comments and whitespace
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Match package and version
            match = re.match(r'^([A-Za-z0-9_.\-]+)(\[[A-Za-z0-9_,\-]+\])?([<>=!~].+)?$', line)
            if match:
                package = match.group(1)
                extras = match.group(2) if match.group(2) else ""
                version_spec = match.group(3) if match.group(3) else ""
                # Include extras in the package name
                full_package = package + extras
                requirements.append((full_package, version_spec))
            else:
                logging.warning(f"Line {line_num}: Skipping unrecognized line: {line}")
    return requirements

def is_pypi_package(package):
    """
    Determines if a package is likely a PyPI-only package.
    This is a heuristic; you may need to adjust based on your specific packages.
    """
    # List of known PyPI-only packages can be extended
    pypi_only_packages = {
        'absl-py', 'aider-chat', 'aiohappyeyeballs', 'aiohttp', 'aiosignal',
        'anyio', 'argon2-cffi', 'argon2-cffi-bindings', 'asttokens',
        'async-lru', 'attrs', 'babel', 'backoff', 'beartype', 'beautifulsoup4',
        'bitsandbytes', 'black', 'bleach', 'blinker', 'bottle', 'build',
        'CacheControl', 'cachetools', 'cleo', 'click', 'colorama', 'colorlog',
        'CoLT5-attention', 'comm', 'commonmark', 'ConfigArgParse',
        'contourpy', 'coverage', 'crashtest', 'decorator', 'defusedxml',
        'diff-match-patch', 'dill', 'diskcache', 'distlib', 'distro',
        'docker-pycreds', 'drawsvg', 'dulwich', 'einops', 'einops-exts',
        'einx', 'entrypoints', 'executing', 'fairscale', 'fastjsonschema',
        'filelock', 'flake8', 'fonttools', 'fqdn', 'frozendict',
        'frozenlist', 'fsspec', 'gitdb', 'GitPython', 'google-ai-generativelanguage',
        'google-api-core', 'google-api-python-client', 'google-auth',
        'google-auth-httplib2', 'google-generativeai', 'googleapis-common-protos',
        'greenlet', 'grep-ast', 'grpcio', 'grpcio-status', 'h11', 'httpcore',
        'httplib2', 'httpx', 'huggingface-hub', 'hypothesis', 'idna',
        'importlib_metadata', 'importlib_resources', 'iniconfig', 'installer',
        'ipykernel', 'ipython', 'isoduration', 'isort', 'jaraco.classes',
        'jedi', 'jeepney', 'Jinja2', 'jiter', 'joblib', 'json5',
        'jsonpointer', 'jsonschema', 'jsonschema-specifications', 'jupyter-events',
        'jupyter-lsp', 'jupyter-server-mathjax', 'jupyter_client',
        'jupyter_core', 'jupyter_server', 'jupyter_server_terminals',
        'jupyterlab', 'jupyterlab_git', 'jupyterlab_pygments',
        'jupyterlab_server', 'keyring', 'kiwisolver', 'libcst', 'lightning',
        'lightning-utilities', 'lion-pytorch', 'litellm', 'local-attention',
        'loguru', 'Mako', 'Markdown', 'markdown-it-py', 'MarkupSafe',
        'matplotlib', 'matplotlib-inline', 'mccabe', 'mdurl',
        'memory-profiler', 'mistune', 'more-itertools', 'mpmath',
        'msgpack', 'multidict', 'multiprocess', 'mypy', 'mypy-extensions',
        'nbclient', 'nbconvert', 'nbdime', 'nbformat', 'nest-asyncio',
        'networkx', 'nltk', 'notebook_shim', 'numpy', 'openai', 'optuna',
        'optuna-dashboard', 'optuna-integration', 'overrides',
        'packaging', 'pandas', 'pandocfilters', 'parso', 'pathspec',
        'pexpect', 'pillow', 'pkginfo', 'platformdirs', 'playwright',
        'plotly', 'pluggy', 'poetry', 'poetry-core', 'poetry-plugin-export',
        'prometheus_client', 'prompt_toolkit', 'proto-plus', 'protobuf',
        'psutil', 'ptyprocess', 'pure_eval', 'pyarrow', 'pyasn1',
        'pyasn1_modules', 'pycodestyle', 'pycparser', 'pydantic',
        'pydantic_core', 'pydeck', 'pydub', 'pyee', 'pyflakes',
        'Pygments', 'pyngrok', 'pynvml', 'pypandoc', 'pyparsing',
        'PyPDF2', 'pyperclip', 'pyproject-api', 'pyproject_hooks',
        'pytest', 'pytest-cov', 'pytest-mock', 'python-dateutil',
        'python-dotenv', 'python-json-logger', 'pytorch-lightning',
        'pytz', 'PyYAML', 'pyzmq', 'RapidFuzz', 'referencing',
        'regex', 'requests', 'requests-toolbelt', 'rfc3339-validator',
        'rfc3986-validator', 'rich', 'rpds-py', 'rsa', 'ruff',
        'safetensors', 'scikit-learn', 'scipy', 'seaborn',
        'SecretStorage', 'Send2Trash', 'sentencepiece', 'sentry-sdk',
        'setproctitle', 'setuptools', 'shellingham', 'six', 'smmap',
        'sniffio', 'sortedcontainers', 'sounddevice', 'soundfile',
        'soupsieve', 'SQLAlchemy', 'stack-data', 'streamlit', 'sympy',
        'tenacity', 'tensorboard', 'tensorboard-data-server',
        'terminado', 'threadpoolctl', 'tiktoken', 'timm', 'tinycss2',
        'tokenizers', 'tokenmonster', 'toml', 'tomlkit', 'toolz',
        'torch', 'torchdiffeq', 'TorchFix', 'torchmetrics',
        'torchsummary', 'torchvision', 'tornado', 'tox', 'tqdm',
        'traitlets', 'transformers', 'tree-sitter', 'tree-sitter-languages',
        'triton', 'trove-classifiers', 'types-python-dateutil',
        'typing', 'typing-inspect', 'typing_extensions', 'tzdata',
        'ultralytics-thop', 'uri-template', 'uritemplate',
        'urllib3', 'vector-quantize-pytorch', 'virtualenv', 'wandb',
        'watchdog', 'wcwidth', 'webcolors', 'webencodings',
        'websocket-client', 'Werkzeug', 'wget', 'xxhash', 'yarl',
        'youtube-transcript-api', 'zetascale', 'zipp'
    }
    return package.lower() in pypi_only_packages

def construct_version_spec(version_spec):
    """
    Constructs the version specifier string for magic add command.
    """
    if not version_spec:
        return ""
    # Magic CLI uses the same version specifiers as pip/conda
    return version_spec

def check_conda_package(package, version_spec):
    """
    Checks if the package exists in Conda repositories with the specified version.
    Returns True if available, False otherwise.
    """
    # Check if 'conda' is available
    try:
        subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        logging.warning("'conda' is not installed or not found in PATH. Falling back to PyPI for package installation.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking Conda version: {e.stderr.strip()}")
        return False

    # Proceed with checking the package in Conda
    try:
        cmd = ["conda", "search", package]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if version_spec:
            # Extract version numbers
            versions = re.findall(r'\s+(\d+\.\d+\.\d+)', result.stdout)
            # Simplistic version matching; can be improved
            for ver in versions:
                # Create a regex pattern based on version_spec
                # This is a simplified approach and may not cover all cases
                pattern = version_spec.replace('>=', r'\d+\.\d+\.\d+')
                pattern = pattern.replace('<=', r'\d+\.\d+\.\d+')
                pattern = pattern.replace('==', r'\d+\.\d+\.\d+')
                pattern = pattern.replace('>', r'\d+\.\d+\.\d+')
                pattern = pattern.replace('<', r'\d+\.\d+\.\d+')
                pattern = pattern.replace('!=', r'\d+\.\d+\.\d+')
                pattern = pattern.replace('~=', r'\d+\.\d+\.\d+')
                pattern = pattern.replace(',', '|')
                if re.match(pattern, ver):
                    return True
            return False
        else:
            versions = re.findall(r'\s+(\d+\.\d+\.\d+)', result.stdout)
            return bool(versions)
    except subprocess.CalledProcessError:
        return False


def add_dependency(package, version_spec, use_pypi=False):
    """
    Constructs and executes the magic add command for a single package.
    Returns True if successful, False otherwise.
    """
    cmd = ["magic", "add"]
    if use_pypi:
        cmd.append("--pypi")
    pkg = package + version_spec
    cmd.append(pkg)
    logging.info(f"Adding package: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(f"Successfully added {pkg}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to add {pkg}. Error: {e.stderr.strip()}")
        return False

def main(requirements_file, pypi_packages=None, parallel=False, max_workers=4):
    """
    Main function to parse requirements and add them to Magic.
    """
    requirements = parse_requirements(requirements_file)
    if not requirements:
        logging.error("No valid requirements found. Exiting.")
        sys.exit(1)

    results = {}

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_package = {}
            for package, version_spec in requirements:
                use_pypi = False
                if pypi_packages and package.lower() in [p.lower() for p in pypi_packages]:
                    use_pypi = True
                else:
                    # Heuristic to determine if the package should be added via PyPI
                    use_pypi = is_pypi_package(package)
                
                # If not using PyPI, check if the package exists in Conda
                if not use_pypi:
                    if not check_conda_package(package, version_spec):
                        logging.warning(f"Package {package} with spec {version_spec} not found in Conda. Falling back to PyPI.")
                        use_pypi = True

                version_str = construct_version_spec(version_spec)
                future = executor.submit(add_dependency, package, version_str, use_pypi)
                future_to_package[future] = package

            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    success = future.result()
                    results[package] = success
                except Exception as exc:
                    logging.error(f"Package {package} generated an exception: {exc}")
                    results[package] = False
    else:
        for package, version_spec in requirements:
            use_pypi = False
            if pypi_packages and package.lower() in [p.lower() for p in pypi_packages]:
                use_pypi = True
            else:
                # Heuristic to determine if the package should be added via PyPI
                use_pypi = is_pypi_package(package)
            
            # If not using PyPI, check if the package exists in Conda
            if not use_pypi:
                if not check_conda_package(package, version_spec):
                    logging.warning(f"Package {package} with spec {version_spec} not found in Conda. Falling back to PyPI.")
                    use_pypi = True

            version_str = construct_version_spec(version_spec)
            success = add_dependency(package, version_str, use_pypi)
            results[package] = success

    # Summary
    logging.info("Dependency addition process completed.")
    success_packages = [pkg for pkg, success in results.items() if success]
    failed_packages = [pkg for pkg, success in results.items() if not success]

    logging.info(f"Successfully added packages ({len(success_packages)}): {', '.join(success_packages)}")
    if failed_packages:
        logging.error(f"Failed to add packages ({len(failed_packages)}): {', '.join(failed_packages)}")
    else:
        logging.info("All packages added successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add dependencies from requirements.txt to Magic with robust error handling.")
    parser.add_argument(
        "requirements",
        type=str,
        help="Path to the requirements.txt file."
    )
    parser.add_argument(
        "--pypi",
        type=str,
        nargs='*',
        default=None,
        help="List of packages to add via PyPI instead of Conda."
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help="Enable parallel addition of packages."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads for parallel execution."
    )
    args = parser.parse_args()
    main(args.requirements, pypi_packages=args.pypi, parallel=args.parallel, max_workers=args.max_workers)
