#!/bin/bash

find src/ include/ -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' | clang-format --style=file -i --files=/dev/stdin
