FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists!*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8888

ENV PYTHONUNBUFFERED=1

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

