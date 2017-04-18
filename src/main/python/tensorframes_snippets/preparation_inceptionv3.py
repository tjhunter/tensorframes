from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time


import numpy as np
import tensorflow as tf

from preprocessing import inception_preprocessing
import datasets.imagenet as imagenet
from nets import inception
import datasets.dataset_utils as dataset_utils

import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.framework import graph_util

slim = tf.contrib.slim

default_image_size = 299


####### Download the network data
# The URL of the checkpointed data.
url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
# The name of the checkpoint file:
checkpoint_file = 'inception_v3.ckpt'
# Specify where you want to download the model to
checkpoints_dir = '/tmp/checkpoints'

checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

if not tf.gfile.Exists(checkpoint_path):
    print('Downloading the model...')
    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

#### TEST TO REMOVE

s = tf.constant("This is string")
r = tf.decode_raw(s, tf.int8)
s2 = tf.as_string(r)
sess = tf.InteractiveSession()
print(s2.eval())

###### Building the computation graph

# All this code can be run once. It assembles the computation graph, fills it with the checkpointed
# coefficients, and then saves it as a protocol buffer description.

# Build the graph
g = tf.Graph()
with g.as_default():
    # Keep for now a placeholder that will eventually be filled with the content of the image.
    # This code only accepts JPEG images, which is the most common image format.
    image_string = tf.placeholder(tf.string, [], name="image_input")

    # Decode string into matrix with intensity values
    image = tf.image.decode_jpeg(image_string, channels=3)

    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    processed_image = inception_preprocessing.preprocess_image(image,
                                                         default_image_size,
                                                         default_image_size,
                                                         is_training=False)

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images  = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.
    # Note: like the Arabian nights, inception defines 1001 classes
    # to include a background class (the first).
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images,
                               num_classes=1001,
                               is_training=False)

    # In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)

    # Just focus on the top predictions
    top_pred = tf.nn.top_k(tf.squeeze(probabilities), k=5, name="top_predictions")

    # These are the outputs we will be requesting from the network.
    output_nodes = [probabilities, top_pred.indices, top_pred.values]

# Create the saver
with g.as_default():
    model_variables = slim.get_model_variables('InceptionV3')
    saver = tf_saver.Saver(model_variables, reshape=False)

def get_op_name(tensor):
    return tensor.name.split(":")[0]

# Export the network
with g.as_default():
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        # The add_shapes option is important: Spark requires this extra shape information to infor the
        # correct types.
        input_graph_def = g.as_graph_def(add_shapes=True)
        output_tensor_names = [node.name for node in output_nodes]
        output_node_names = [n.split(":")[0] for n in output_tensor_names]
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names,
            variable_names_blacklist=[])

# The variable 'output_graph_def' now contains all the description of the computation.
# The variables in the 'output_nodes' list will be used to know what to output.

####### Testing the computation graph

# This code performs a sanity check, by running the network against some image content downloaded from the internet.

g2 = tf.Graph()
with g2.as_default():
    tf.import_graph_def(output_graph_def, name='')

#### Download an image
import requests

# Example picture:
# Specify where you want to download the model to
images_dir = '/tmp/image_data'

if not tf.gfile.Exists(images_dir):
    tf.gfile.MakeDirs(images_dir)

image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/WeaverAntDefense.JPG/640px-WeaverAntDefense.JPG'
image_url = 'https://www.tensorflow.org/images/cropped_panda.jpg'
image_path = os.path.join(images_dir, image_url.split('/')[-1])

if not tf.gfile.Exists(image_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_path, 'wb') as f:
            f.write(response.content)

image_data = tf.gfile.FastGFile(image_path, 'rb').read()

with g2.as_default():
    input_node2 = g2.get_operation_by_name(get_op_name(image))
    output_nodes2 = [g2.get_tensor_by_name(n) for n in output_tensor_names]
    with tf.Session() as sess:
        (probabilities_, indices_, values_) = sess.run(output_nodes2, {'image_input:0':image_data})

names = imagenet.create_readable_names_for_imagenet_labels()
for i in range(5):
    index = indices_[i]
    print('Probability %d %0.2f => [%s]' % (index, values_[i], names[index]))


###### Perform some evaluation with TensorFrames

# This code takes the network and a directory that contains some image content. It shows how to process the content
# using Spark dataframes and Tensorframes.

import tensorframes as tfs
sc.setLogLevel('INFO')

raw_images_miscast = sc.binaryFiles("file:"+images_dir)
raw_images = raw_images_miscast.map(lambda x: (x[0], bytearray(x[1])))

df = spark.createDataFrame(raw_images).toDF('image_uri', 'image_data')
df

with g2.as_default():
    index_output = tf.identity(g2.get_tensor_by_name('top_predictions:1'), name="index")
    value_output = tf.identity(g2.get_tensor_by_name('top_predictions:0'), name="value")
    pred_df = tfs.map_rows([index_output, value_output], df, feed_dict={'image_input':'image_data'})

pred_df.select('index', 'value').head()

