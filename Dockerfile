FROM langchain/langgraph-api:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev python3 ca-certificates python3-numpy python3-setuptools python3-wheel python3-pip g++ gcc ninja-build cmake build-essential autoconf automake libtool libmagic-dev poppler-utils libreoffice libomp-dev tesseract-ocr tesseract-ocr-por libyaml-dev ffmpeg libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git libpq5 libpq-dev libxml2-dev libxslt1-dev libcairo2-dev libgirepository1.0-dev libgraphviz-dev libjpeg-dev libopencv-dev libpango1.0-dev libprotobuf-dev protobuf-compiler rustc cargo libwebp-dev libzbar0 libzbar-dev imagemagick ghostscript pandoc aria2 zsh bash-completion libpq-dev pkg-config libssl-dev  openssl unzip gzip vim tree less sqlite3 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.11.2
RUN echo "source $HOME/.asdf/asdf.sh" >> ~/.bashrc
ENV TAPLO_VERSION=0.9.3
COPY ./install_taplo.sh .
RUN chmod +x install_taplo.sh && bash -x ./install_taplo.sh && mv taplo /usr/local/bin/taplo && rm install_taplo.sh
# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
# Install justfile
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/bin
# Install UV 0.5.14
ADD https://astral.sh/uv/0.5.14/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"
# Configure UV
ENV UV_SYSTEM_PYTHON=1
ENV UV_PIP_DEFAULT_PYTHON=/usr/bin/python3
ENV UV_LINK_MODE=copy
# Compiling Python source files to bytecode is typically desirable for production images as it tends to improve startup time (at the cost of increased installation time).
# ENV UV_COMPILE_BYTECODE=1
ENV UV_CACHE_DIR=/root/.cache/uv/
# Install dependencies first (for better caching)
WORKDIR /deps/democracy-exe
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=bind,source=uv.lock,target=uv.lock --mount=type=bind,source=pyproject.toml,target=pyproject.toml uv sync --frozen --no-install-project  --all-extras  --verbose --no-dev
# Copy project and install
COPY . /deps/democracy-exe
RUN --mount=type=cache,target=/root/.cache/uv uv sync --verbose --no-dev --frozen && uv tool dir --bin
# Pre-compile bytecode
# RUN python3 -m compileall .
RUN ls -lta && pwd && ls -lta /deps && tree /deps
# Use the virtual environment automatically
# ENV VIRTUAL_ENV="/deps/democracy-exe/.venv"
# uv: Once the project is installed, you can either activate the project virtual environment by placing its binary directory at the front of the path:
# Place entry points in the environment at the front of the path
# ENV PATH="/deps/democracy-exe/.venv/bin:$PATH"

ADD . /deps/democracy-exe

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"memgraph": "/deps/democracy-exe/democracy_exe/agentic/graph.py:memgraph"}'

WORKDIR /deps/democracy-exe
