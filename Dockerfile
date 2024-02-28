ARG PYTHON_IMAGE
ARG PYTHON_IMAGE_VERSION

FROM ${PYTHON_IMAGE}:${PYTHON_IMAGE_VERSION}

ARG POETRY_VERSION

RUN pip install --upgrade pip setuptools
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /opt/pode

COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install

COPY README.md .
COPY pode/ pode/
COPY tests/ tests/
