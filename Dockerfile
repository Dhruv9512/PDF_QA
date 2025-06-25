# Base image with Conda
FROM continuumio/miniconda3

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/envs/myenv/bin:$PATH
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN conda create -n myenv python=3.11 && \
    /opt/conda/envs/myenv/bin/pip install --upgrade pip && \
    /opt/conda/envs/myenv/bin/pip install -r requirements.txt

COPY . .
COPY start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

CMD ["/bin/bash", "/app/start-all.sh"]
