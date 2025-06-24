# Base image with Conda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Environment setup
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH /opt/conda/envs/myenv/bin:$PATH
ENV PYTHONPATH=/app

# Copy and install dependencies
COPY requirements.txt .
RUN conda create -n myenv python=3.11 && \
    /opt/conda/envs/myenv/bin/pip install --upgrade pip && \
    /opt/conda/envs/myenv/bin/pip install -r requirements.txt

# Copy the full app
COPY . .

# Default command
CMD ["conda", "run", "--no-capture-output", "-n", "myenv", "gunicorn", "PDF_QA.wsgi", "--workers", "3", "--worker-class", "gevent", "--timeout", "120", "--bind", "0.0.0.0:8000"]
