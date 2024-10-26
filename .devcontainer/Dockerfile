# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12.1
RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev

# Set Python3.12 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install Magic
RUN curl -ssL https://magic.modular.com/00cdab34-48af-46d3-a2a1-f8861e24a51c | bash

# Add Magic to PATH
ENV PATH="/home/codespace/.modular/bin:${PATH}"

# Initialize Magic project directory (optional)
# RUN magic init gpt-arc

# Install MAX
RUN magic add max >=24.5.0,<25

# Install Mojo
# Assuming `mojo` can be installed via Magic or another method
# Replace the following lines with the actual installation steps for Mojo
# For example:
# RUN magic add mojo

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Define any additional setup or environment variables here
