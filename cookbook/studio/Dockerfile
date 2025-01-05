FROM langchain/langgraph-api:3.11



ADD requirements.txt /deps/__outer_studio/src/requirements.txt
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -r /deps/__outer_studio/src/requirements.txt

ADD . /deps/__outer_studio/src
RUN set -ex && \
    for line in '[project]' \
                'name = "studio"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]'; do \
        echo "$line" >> /deps/__outer_studio/pyproject.toml; \
    done

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"parallelization": "/deps/__outer_studio/src/parallelization.py:graph", "sub_graphs": "/deps/__outer_studio/src/sub_graphs.py:graph", "map_reduce": "/deps/__outer_studio/src/map_reduce.py:graph", "research_assistant": "/deps/__outer_studio/src/research_assistant.py:graph"}'

WORKDIR /deps/__outer_studio/src
