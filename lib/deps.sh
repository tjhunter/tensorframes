#!/usr/bin/env bash
rm -rf ./jni
rm -rf libtensorflow*jar
export TF_VERSION=1.1.0-rc0
mkdir -p ./jni
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-$TF_VERSION.jar > libtensorflow-$TF_VERSION.jar
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-src-$TF_VERSION.jar > libtensorflow-src-$TF_VERSION.jar
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-$TF_VERSION.tar.gz > libtensorflow_jni-cpu-linux-x86_64-$TF_VERSION.tar.gz
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-$TF_VERSION.tar.gz > libtensorflow_jni-cpu-darwin-x86_64-$TF_VERSION.tar.gz
cat libtensorflow_jni-cpu-darwin-x86_64-$TF_VERSION.tar.gz | tar -xz -C ./jni
cat libtensorflow_jni-cpu-linux-x86_64-$TF_VERSION.tar.gz | tar -xz -C ./jni
