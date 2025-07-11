#!/bin/bash

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -B build .\
      -DFAISS_ENABLE_PYTHON=OFF \
      -DFAISS_ENABLE_GPU=OFF \
      -DBUILD_TESTING=OFF

cmake --build build -- -j$(nproc)

mkdir -p sift

if [ ! -d "./sift/sift10k" ]; then
  wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
  tar -xvf siftsmall.tar.gz
  rm siftsmall.tar.gz
  mv siftsmall sift/sift10k
fi

if [ ! -d "./sift/sift1M" ]; then
  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  mkdir -p temp_extract
  tar -xvf sift.tar.gz -C temp_extract
  rm sift.tar.gz
  mv temp_extract/sift sift/sift1M
  rmdir temp_extract
fi

if [ ! -f "./compile_commands.json" ]; then
  ln -s ./build/compile_commands.json .
fi
