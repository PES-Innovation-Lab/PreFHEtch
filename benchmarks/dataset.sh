#!/bin/bash

# run this from root
# if dataset does not exist pull it
if [ ! -d "sift/siftsmall/" ]; then
  wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
  tar -xvf siftsmall.tar.gz
  rm siftsmall.tar.gz
  mkdir sift 
  mv siftsmall sift/
fi
