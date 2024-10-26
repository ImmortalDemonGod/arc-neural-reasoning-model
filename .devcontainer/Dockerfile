# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies and Python 3.12
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        build-essential \
        curl \
        git \
        zsh \
        ca-certificates \
        fonts-powerline \
        pip \
        python3-pip && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3.12-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    rm -rf /var/lib/apt/lists/*

# Install and configure Oh My Zsh with Powerlevel10k
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    git clone https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k && \
    sed -i 's/ZSH_THEME=".*"/ZSH_THEME="powerlevel10k\/powerlevel10k"/g' ~/.zshrc

# Install Magic CLI
RUN curl -ssL https://magic.modular.com/00cdab34-48af-46d3-a2a1-f8861e24a51c | bash
ENV PATH="/root/.modular/bin:${PATH}"

# Set the OPENAI_API_KEY environment variable
ENV OPENAI_API_KEY="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"

# Set the default shell to zsh
SHELL ["/bin/zsh", "-c"]