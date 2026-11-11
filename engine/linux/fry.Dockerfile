#the engine as the CCC builds and runs its engines (github.com/ChessCom/ccc-configs): no gpu at the build, no network at the run
FROM nvidia/cuda:12.9.2-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN apt-get update && \
    apt-get install -y git curl ca-certificates build-essential && \
    apt-get clean

#the tag's code, for which the start-up caches are made: build.sh finds no gpu here and downloads them; the build's own tools go
#once it is done
ARG CACHE_BUST
RUN git clone --branch ccc-2026-11-11 https://github.com/GilliamWong/chess-transformer && \
    cd chess-transformer && \
    engine/linux/build.sh && \
    rm -rf /root/.cache/uv /root/.cargo /root/.rustup

CMD [ "/root/chess-transformer/engine/linux/engine.sh" ]
