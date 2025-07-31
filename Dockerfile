FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    git build-essential curl uuid-dev \
    libjsoncpp-dev libprotobuf-dev protobuf-compiler pkg-config zlib1g-dev \
    libssl-dev libblas-dev liblapack-dev \
    libcurl4-openssl-dev \
    wget && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz | tar --strip-components=1 -xz -C /usr/local

RUN git clone --depth=1 --recurse-submodules https://github.com/drogonframework/drogon.git && \
    mkdir drogon/build && cd drogon/build && \
    cmake .. && \
    make -j$(nproc) && make install && \
    cd ../.. && rm -rf drogon

RUN git clone --depth=1 https://github.com/libcpr/cpr.git && \
    mkdir cpr/build && cd cpr/build && \
    cmake .. -DBUILD_CPR_TESTS=OFF -DCMAKE_USE_OPENSSL=ON -DCPR_USE_SYSTEM_CURL=ON && \
    make -j$(nproc) && make install && \
    cd ../.. && rm -rf cpr

WORKDIR /PreFHEtch

COPY . .

RUN rm -rf build

RUN cmake -B build .
RUN cmake --build build -- -j$(nproc)

CMD ["/bin/bash"]
