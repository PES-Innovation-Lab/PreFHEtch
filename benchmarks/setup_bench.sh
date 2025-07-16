#!/bin/bash

setup() {
  sudo apt-get update && sudo apt-get install -y \
    git build-essential curl uuid-dev \
    libjsoncpp-dev pkg-config zlib1g-dev \
    libssl-dev libblas-dev liblapack-dev \
    libcurl4-openssl-dev \
    wget && sudo apt-get clean &&
    sudo rm -rf /var/lib/apt/lists/*

  curl -sSL https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz | sudo tar --strip-components=1 -xz -C /usr/local

  git clone --depth=1 --recurse-submodules https://github.com/drogonframework/drogon.git &&
    mkdir drogon/build && cd drogon/build &&
    cmake .. &&
    make -j$(nproc) && sudo make install &&
    cd ../.. && rm -rf drogon

  git clone --depth=1 https://github.com/libcpr/cpr.git &&
    mkdir cpr/build && cd cpr/build &&
    cmake .. -DBUILD_CPR_TESTS=OFF -DCMAKE_USE_OPENSSL=ON -DCPR_USE_SYSTEM_CURL=ON &&
    make -j$(nproc) && sudo make install &&
    cd ../.. && rm -rf cpr

  cd .. && rm -rf build && cmake -B build . && cmake --build build && cd benchmarks

}

uninstall() {
  sudo rm -rf /usr/local/lib/{libcpr.a,libdrogon.a,libtrantor.a}
  sudo rm -rf /usr/local/lib/cmake
}

sift() {
  cd .. && mkdir sift && cd sift &&
    wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz &&
    tar -xzf siftsmall.tar.gz &&
    cd ..
}

case "$1" in
setup)
  setup
  ;;
clean)
  uninstall
  ;;
sift)
  sift
  ;;
*)
  echo "Invalid option"
  ;;
esac
