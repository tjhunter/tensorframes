#!/usr/bin/env bash
mkdir -p ./jni
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.0.0-PREVIEW1.jar > ibtensorflow-1.0.0-PREVIEW1.jar
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-src-1.0.0-PREVIEW1.jar > libtensorflow-src-1.0.0-PREVIEW1.jar
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.0.0-PREVIEW1.tar.gz > libtensorflow_jni-cpu-linux-x86_64-1.0.0-PREVIEW1.tar.gz
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.0.0-PREVIEW1.tar.gz > libtensorflow_jni-cpu-darwin-x86_64-1.0.0-PREVIEW1.tar.gz
cat libtensorflow_jni-cpu-darwin-x86_64-1.0.0-PREVIEW1.tar.gz | tar -xz -C ./jni
cat libtensorflow_jni-cpu-linux-x86_64-1.0.0-PREVIEW1.tar.gz | tar -xz -C ./jni
