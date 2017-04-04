import numpy as np
import tensorframes as tfs
import tensorflow as tf
sc.setLogLevel('INFO')

with tf.gfile.FastGFile('/tmp/inception/inception_v3.ckpt', 'rb') as f:
    model_data = f.read()

g = tf.Graph()
with g.as_default():
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model_data)
    _ = tf.import_graph_def(graph_def, name='')

raw_images_miscast = sc.binaryFiles("file:/tmp/101_ObjectCategories/BACKGROUND_Google") # file:
raw_images = raw_images_miscast.map(lambda x: (x[0], bytearray(x[1])))

df = spark.createDataFrame(raw_images).toDF('image_uri', 'image_data')
df

with g.as_default() as _:
    pred_output = g.get_tensor_by_name('softmax:0')
    pred_df = tfs.map_rows(pred_output, df, feed_dict={'DecodeJpeg/contents':'image_data'})


## TO RUN FROM THE SLIM DIRECTORY

import datasets as datasets
import datasets.dataset_utils as dataset_utils

import datasets.imagenet as imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.tools.freeze_graph import freeze_graph
import urllib3
import os

url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

# Specify where you want to download the model to
checkpoints_dir = '/tmp/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

slim = tf.contrib.slim

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = vgg.vgg_16.default_image_size




from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib

def get_op_name(tensor):
    return tensor.name.split(":")[0]

# Build the graph
g = tf.Graph()
with g.as_default():
    # Open specified url and load image as a string
    image_string = open("/tmp/101_ObjectCategories/ant/image_0001.jpg", 'rb').read()

    # Decode string into matrix with intensity values
    image = tf.image.decode_jpeg(image_string, channels=3)

    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images  = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)

    # In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)

    # Just focus on the top predictions
    top_pred = tf.nn.top_k(tf.squeeze(probabilities), 5, name="top_predictions")

# Initialize the values
with g.as_default():

    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')
    model_variables = slim.get_model_variables('vgg_16')
    saver = tf_saver.Saver(model_variables, reshape=False)

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        # Load weights
        #init_fn(sess)

        # We want to get predictions, image as numpy matrix
        # and resized and cropped piece that is actually
        # being fed to the network.
        output_nodes = [probabilities, top_pred.indices, top_pred.values]
        probabilities_, indices_, values_ = sess.run(output_nodes)
        probabilities_ = probabilities_[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities_),
                                            key=lambda x:x[1])]

with g.as_default():
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        input_graph_def = g.as_graph_def()
        output_tensor_names = [node.name for node in output_nodes]
        output_node_names = [n.split(":")[0] for n in output_tensor_names]
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names,
            variable_names_blacklist=[])

g2 = tf.Graph()
with g2.as_default():
    tf.import_graph_def(output_graph_def, name='')

image_data = tf.gfile.FastGFile("/tmp/101_ObjectCategories/ant/image_0001.jpg", 'rb').read()

with g2.as_default():
    input_node2 = g2.get_operation_by_name(get_op_name(image))
    output_nodes2 = [g2.get_tensor_by_name(n) for n in output_tensor_names]
    with tf.Session() as sess:
        (probabilities_, indices_, values_) = sess.run(output_nodes2, {'DecodeJpeg/contents:0':image_data})



names = imagenet.create_readable_names_for_imagenet_labels()
for i in range(5):
    index = sorted_inds[i]
    # Now we print the top-5 predictions that the network gives us with
    # corresponding probabilities. Pay attention that the index with
    # class names is shifted by 1 -- this is because some networks
    # were trained on 1000 classes and others on 1001. VGG-16 was trained
    # on 1000 classes.
    print('Probability %d %0.2f => [%s]' % (index, probabilities_[index], names[index+1]))
