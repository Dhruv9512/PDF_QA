# Base image with Conda
FROM continuumio/miniconda3

# Install netcat for Redis check
RUN apt-get update && apt-get install -y netcat && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/envs/myenv/bin:$PATH
ENV PYTHONPATH=/app

# Copy and install dependencies
COPY requirements.txt .
RUN conda create -n myenv python=3.11 && \
    /opt/conda/envs/myenv/bin/pip install --upgrade pip && \
    /opt/conda/envs/myenv/bin/pip install -r requirements.txt

# Copy your app and start script
COPY . .
COPY start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

# Run Gunicorn + Celery in one container
CMD ["/bin/bash", "/app/start-all.sh"]
