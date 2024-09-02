from setuptools import setup, find_packages

setup(
    name="arc-neural-reasoning-model",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=20.8b1",
            "isort>=5.0",
            "flake8>=3.9",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A neural reasoning model for the ARC challenge",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/arc-neural-reasoning-model",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
