FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 python3.9-distutils \
    curl git wget \
    libgl1 libglu1-mesa libgl1-mesa-glx libgl1-mesa-dri \
    libegl1 libosmesa6 xvfb ffmpeg \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install notebook jupyterlab ipykernel

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=7000", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]