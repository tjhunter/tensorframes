#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: ./update-tf-proto.sh TF_VERSION"
  exit 1
fi

version=$1

mkdir -p tf_tmp
rm -rf tf_tmp/*
cd tf_tmp
curl -L https://codeload.github.com/tensorflow/tensorflow/zip/v$version -o tensorflow-$version.zip
unzip tensorflow-$version.zip
rm -f ../../src/main/protobuf/tensorflow/core/framework/*.proto
cp tensorflow-$version/tensorflow/core/framework/*.proto ../../src/main/protobuf/tensorflow/core/framework/
cd ..
rm -rf tf_tmp
