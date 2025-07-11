#!/bin/bash

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug -B build . -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF
cmake --build build -- -j$(nproc)


if [ ! -d "./dataset" ]; then
  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar -xvf sift.tar.gz
  rm sift.tar.gz
  mkdir -p sift
  mv sift dataset/sift1M
fi
