FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
# Build: docker build -t mflux-server-cuda .
# Run: docker run --rm --gpus all -p 4030:4030 mflux-server-cuda
# Note: install NVIDIA Container Toolkit on the host (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run docker with --gpus all.

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::Check-Valid-Until=false \
    && mkdir -p /tmp/apt-cache \
    && apt-get install -y --no-install-recommends -o Dir::Cache::archives=/tmp/apt-cache \
        ca-certificates \
        gnupg \
        ubuntu-keyring \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-cache

RUN apt-get update \
    && mkdir -p /tmp/apt-cache \
    && apt-get install -y --no-install-recommends -o Dir::Cache::archives=/tmp/apt-cache \
        software-properties-common \
        curl \
        git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends -o Dir::Cache::archives=/tmp/apt-cache \
        python3.12 \
        python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-cache

COPY requirements_cuda.txt /app/requirements_cuda.txt
RUN python3.12 -m pip install --no-cache-dir --ignore-installed -r /app/requirements_cuda.txt

COPY . /app

EXPOSE 4030

CMD ["python3.12", "server_cuda.py", "--host", "0.0.0.0", "--device", "cuda", "--workers", "1", "--device_map", "balanced"]
