# Use docker-compose to build and run this container.
# $ docker-compose build [--build-arg PYTHON_VERSION=2.7]
# $ docker-compose up -d
# $ docker-compose exec master bash -i -c "..."
# $ docker-compose down

FROM ubuntu:16.04

ARG PYTHON_VERSION=3.6

RUN apt-get update && \
    apt-get install -y wget bzip2 openjdk-8-jdk unzip && \
    apt-get clean

# Install Miniconda.
# Reference: https://hub.docker.com/r/continuumio/miniconda/~/dockerfile/
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Create tensorframes conda env.
ENV PYTHON_VERSION $PYTHON_VERSION
COPY ./environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda create -n tensorframes python=$PYTHON_VERSION && \
    /opt/conda/bin/conda env update -n tensorframes -f /tmp/environment.yml && \
    echo "conda activate tensorframes" >> ~/.bashrc

# Install Spark and update env variables.
ENV SCALA_BINARY_VERSION 2.11.8
ENV SPARK_VERSION 2.4.3
ENV SPARK_BUILD "spark-${SPARK_VERSION}-bin-hadoop2.7"
ENV SPARK_BUILD_URL "http://www.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_BUILD}.tgz"
RUN wget --quiet $SPARK_BUILD_URL -O /tmp/spark.tgz && \
    tar -C /opt -xf /tmp/spark.tgz && \
    mv /opt/$SPARK_BUILD /opt/spark && \
    rm /tmp/spark.tgz
ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH
ENV PYTHONPATH /opt/spark/python/lib/py4j-0.10.7-src.zip:/opt/spark/python/lib/pyspark.zip:$PYTHONPATH
ENV PYSPARK_PYTHON python

# The tensorframes dir will be mounted here.
VOLUME /mnt/tensorframes
WORKDIR /mnt/tensorframes

CMD /bin/bash
