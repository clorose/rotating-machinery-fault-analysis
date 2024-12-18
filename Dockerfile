FROM python:3.11-slim

# Install essential packages
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  git \
  tzdata \ 
  nano \
  zsh \
  htop \
  tree \
  && rm -rf /var/lib/apt/lists/* \
  && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
  && git clone https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
  && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
  && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Copy zsh configuration files
COPY ./zsh/zshrc /root/.zshrc
COPY ./zsh/p10k.zsh /root/.p10k.zsh
COPY ./zsh/aliases.zsh /root/.aliases.zsh

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  . $HOME/.profile && \
  export PATH="$HOME/.local/bin:$PATH" && \
  uv venv /app/.venv && \
  . /app/.venv/bin/activate && \
  uv pip install --upgrade pip

# Set working directory and create directories
WORKDIR /app
RUN mkdir -p /app/data /app/src /app/runs

# Install Python packages
COPY requirements.txt .
RUN . /app/.venv/bin/activate && \
  export PATH="$HOME/.local/bin:$PATH" && \
  uv pip install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/app \
  VIRTUAL_ENV=/app/.venv \
  PATH="$VIRTUAL_ENV/bin:$PATH" \
  TZ=Asia/Seoul

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set zsh as default shell
SHELL ["/bin/zsh", "-c"]