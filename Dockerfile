FROM langchain/langgraph-api:3.12

# Install system dependencies
ENV UV_SYSTEM_PYTHON=1 \
    UV_PIP_DEFAULT_PYTHON=/usr/bin/python3 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv/ \
    PYTHONASYNCIODEBUG=1 \
    DEBIAN_FRONTEND=noninteractive \
    TAPLO_VERSION=0.9.3 \
    UV_VERSION=0.5.16 \
    PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get -qq install -y --no-install-recommends python3-dev python3 ca-certificates python3-numpy python3-setuptools python3-wheel python3-pip g++ gcc ninja-build cmake build-essential autoconf automake libtool libmagic-dev poppler-utils libreoffice libomp-dev tesseract-ocr tesseract-ocr-por libyaml-dev ffmpeg libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git libpq5 libpq-dev libxml2-dev libxslt1-dev libcairo2-dev libgirepository1.0-dev libgraphviz-dev libjpeg-dev libopencv-dev libpango1.0-dev libprotobuf-dev protobuf-compiler rustc cargo libwebp-dev libzbar0 libzbar-dev imagemagick ghostscript pandoc aria2 zsh bash-completion libpq-dev pkg-config libssl-dev  openssl unzip gzip vim tree less sqlite3 && rm -rf /var/lib/apt/lists/*
# debugging, show the current directory and the contents of the deps directory, look for .venv which should not exist.
# RUN ls -lta && echo `pwd` && ls -lta && tree && echo "PATH='/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'" >> ~/.bashrc && echo "PATH='/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'" >> ~/.profile


# Hopefully this will fix the path issue with langgraph studio grabbing the env vars from my host machine.

# Install justfile and taplo
COPY ./install_taplo.sh .
RUN chmod +x install_taplo.sh && bash ./install_taplo.sh && mv taplo /usr/local/bin/taplo && rm install_taplo.sh && curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin


# Install rust


RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | env PATH='/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' bash -s -- -y
ENV PATH='/root/.cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


# Install UV ${UV_VERSION}


ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN env PATH='/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' bash /uv-installer.sh && rm /uv-installer.sh

ENV PATH='/root/.local/bin:/root/.cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'

# Install dependencies first (for better caching)

WORKDIR /deps/democracy-exe


COPY pyproject.toml uv.lock ./


RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=democracy_exe/requirements.txt,target=requirements.txt \
    pip install -U pip && \
    uv sync --frozen --no-install-project --no-dev && \
    PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -r requirements.txt -e . /deps/*


# Copy project project files and install
COPY . /deps/democracy-exe

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

ADD . /deps/democracy-exe

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"react": "/deps/democracy-exe/democracy_exe/agentic/workflows/react/graph.py:graph"}'

WORKDIR /deps/democracy-exe
CMD bash -l
