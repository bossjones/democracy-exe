FROM langchain/langgraph-api:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev python3 ca-certificates python3-numpy python3-setuptools python3-wheel python3-pip g++ gcc ninja-build cmake build-essential autoconf automake libtool libmagic-dev poppler-utils libreoffice libomp-dev tesseract-ocr tesseract-ocr-por libyaml-dev ffmpeg libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git libpq5 libpq-dev libxml2-dev libxslt1-dev libcairo2-dev libgirepository1.0-dev libgraphviz-dev libjpeg-dev libopencv-dev libpango1.0-dev libprotobuf-dev protobuf-compiler rustc cargo libwebp-dev libzbar0 libzbar-dev imagemagick ghostscript pandoc aria2 zsh bash-completion libpq-dev pkg-config libssl-dev  openssl unzip gzip vim && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.11.2
RUN echo "source $HOME/.asdf/asdf.sh" >> ~/.bashrc
ENV TAPLO_VERSION 0.9.3
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
ENV UV_COMPILE_BYTECODE=1
# ENV UV_CACHE_DIR=/opt/uv-cache/
ENV UV_CACHE_DIR=/root/.cache/uv/
# Install dependencies first (for better caching)
WORKDIR /deps/__outer_studio/src
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=bind,source=uv.lock,target=uv.lock --mount=type=bind,source=pyproject.toml,target=pyproject.toml uv sync --frozen --no-install-project  --all-extras  --verbose
# Copy project and install
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --verbose --frozen && uv tool dir --bin
# Pre-compile bytecode
RUN python3 -m compileall .
RUN ls -lta && pwd && ls -lta /deps

ADD democracy_exe/requirements.txt /deps/__outer_democracy_exe/democracy_exe/requirements.txt
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -r /deps/__outer_democracy_exe/democracy_exe/requirements.txt

ADD ./democracy_exe /deps/__outer_democracy_exe/democracy_exe
RUN set -ex && \
    for line in '[project]' \
                'name = "democracy_exe"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]'; do \
        echo "$line" >> /deps/__outer_democracy_exe/pyproject.toml; \
    done

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"memgraph": "/deps/__outer_democracy_exe/democracy_exe/agentic/graph.py:memgraph"}'
