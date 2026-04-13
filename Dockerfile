# Build stage
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libeigen3-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY lookuptable.csv .

RUN mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc)

# Runtime stage
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-highgui4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    python3 \
    python3-pip \
    && pip3 install --no-cache-dir matplotlib \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/bin/ bin/
COPY --from=builder /app/lookuptable.csv .
COPY scripts/ scripts/
COPY experiments/ experiments/
COPY core/ core/

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["bash"]
