ARG BASE_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.1
ARG ETCD_IMAGE=quay.io/coreos/etcd:v3.5.21
ARG NATS_IMAGE=nats:2.11.8

FROM ${ETCD_IMAGE} AS etcd
FROM ${NATS_IMAGE} AS nats
FROM ${BASE_IMAGE}

USER root

COPY deploy/taiji-cluster/ubuntu24_sources.list /etc/apt/sources.list

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim \
        unzip \
        openssh-server \
        openssh-client \
        cron \
        nfs-common \
        lrzsz \
        tmux \
        iputils-ping \
        apache2-utils \
        net-tools \
        ethtool \
        rsync \
        cuda-nsight-systems-13-0 \
        lsof \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
RUN python3 -m pip install --no-cache-dir -i ${PIP_INDEX_URL} pandas matplotlib

# ---------------------------------------------------------------------------
# Build dependencies for Mooncake and dynamo.
# ---------------------------------------------------------------------------

# Mooncake's system packages, taken from Mooncake/dependencies.sh, plus
# libclang-dev: dynamo's nixl-sys runs bindgen, which needs libclang even on its
# stub path (this image ships NIXL's .so files but no headers).
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        pkg-config \
        patchelf \
        libc6-dev \
        libc-bin \
        libibverbs-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libjsoncpp-dev \
        libunwind-dev \
        libnuma-dev \
        libpython3-dev \
        libboost-all-dev \
        libssl-dev \
        libgrpc-dev \
        libgrpc++-dev \
        libprotobuf-dev \
        libyaml-cpp-dev \
        protobuf-compiler-grpc \
        libcurl4-openssl-dev \
        libhiredis-dev \
        liburing-dev \
        libjemalloc-dev \
        libmsgpack-dev \
        libzstd-dev \
        libasio-dev \
        libxxhash-dev \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# yalantinglibs: consumed via find_package, so installing it system-wide is
# enough and the source is dropped. Pinned to Mooncake's submodule tag.
ARG YALANTINGLIBS_TAG=v0.5.7
RUN git clone --depth 1 --branch ${YALANTINGLIBS_TAG} \
        https://github.com/alibaba/yalantinglibs.git /tmp/yalantinglibs \
    && cmake -S /tmp/yalantinglibs -B /tmp/yalantinglibs/build -GNinja \
        -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF \
    && cmake --build /tmp/yalantinglibs/build -j"$(nproc)" \
    && cmake --install /tmp/yalantinglibs/build \
    && ldconfig \
    && rm -rf /tmp/yalantinglibs

# pybind11: consumed by add_subdirectory(extern/pybind11), so it cannot be
# installed system-wide. Staged here as a fallback for when Mooncake's submodule
# is not checked out. Pinned to the commit that submodule records.
# ln -sfn /opt/pybind11 Mooncake/extern/pybind11
ARG PYBIND11_COMMIT=58c382a8e3d7081364d2f5c62e7f429f0412743b
RUN git clone --filter=blob:none https://github.com/pybind/pybind11.git /opt/pybind11 \
    && git -C /opt/pybind11 checkout --detach ${PYBIND11_COMMIT} \
    && rm -rf /opt/pybind11/.git

# Rust toolchain for dynamo, version pinned to match rust-toolchain.toml.
# CARGO_HOME/RUSTUP_HOME are explicit because this base image has no HOME=/root
# (`USER root` does not set it), so rustup's $HOME/.cargo default is unknowable.
ARG RUST_VERSION=1.93.1
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --default-toolchain ${RUST_VERSION} \
    && chmod -R a+rX ${RUSTUP_HOME} ${CARGO_HOME}
RUN rustc --version && cargo --version

# maturin builds dynamo's Python bindings; patchelf extra is what fixes up the
# wheel's rpath.
RUN python3 -m pip install --no-cache-dir -i ${PIP_INDEX_URL} 'maturin[patchelf]'

# ---------------------------------------------------------------------------
# Build Mooncake and dynamo here, not on the GPU host: its source lives on
# shared storage, where compiling is far slower than downloading.
# Sources come from two extra build contexts, so this image must be built as:
#
#   docker buildx build \
#       --build-context mooncake=/data/home/kiryli/Mooncake \
#       --build-context dynamo-src=/data/home/kiryli/dynamo-dev \
#       -f deploy/taiji-cluster/Dockerfile -t <tag> .
#
# Only the artifacts are kept — the trees are deleted. Both stay overridable on
# the host: a later `make install` or `pip install` replaces what is here.
# ---------------------------------------------------------------------------

# USE_CUDA and everything that forces it (USE_MNNVL, USE_NVMEOF) default to OFF,
# so no CUDAToolkit is needed. yalantinglibs is already installed system-wide;
# extern/pybind11 has to sit in the tree, hence the copy from /opt/pybind11.
COPY --from=mooncake . /opt/build/Mooncake
RUN cd /opt/build/Mooncake \
    && ([ -f extern/pybind11/CMakeLists.txt ] \
        || cp -r /opt/pybind11/. extern/pybind11/) \
    && mkdir -p build && cd build \
    && cmake .. && make -j"$(nproc)" && make install && ldconfig \
    && cd / && rm -rf /opt/build/Mooncake

# The wheel is installed, not kept: pip owns dynamo._core from here on, and a
# later `pip install` on the host replaces it.
COPY --from=dynamo-src . /opt/build/dynamo-dev
RUN cd /opt/build/dynamo-dev/lib/bindings/python \
    && maturin build --release --out /tmp/dynwheels \
    && pip install --force-reinstall --no-deps \
        /tmp/dynwheels/ai_dynamo_runtime-*.whl \
    && cd / && rm -rf /opt/build/dynamo-dev /tmp/dynwheels

COPY --from=etcd /usr/local/bin/etcd /usr/local/bin/etcd
COPY --from=etcd /usr/local/bin/etcdctl /usr/local/bin/etcdctl
COPY --from=nats /nats-server /usr/local/bin/nats-server

COPY examples/backends/vllm /opt/dynamo/examples/backends/vllm
COPY deploy/taiji-cluster/start-head.sh /opt/dynamo/start-head.sh
COPY deploy/taiji-cluster/start-worker.sh /opt/dynamo/start-worker.sh

RUN chmod 0755 \
    /usr/local/bin/etcd \
    /usr/local/bin/etcdctl \
    /usr/local/bin/nats-server \
    /opt/dynamo/start-head.sh \
    /opt/dynamo/start-worker.sh
WORKDIR /opt/dynamo/examples/backends/vllm