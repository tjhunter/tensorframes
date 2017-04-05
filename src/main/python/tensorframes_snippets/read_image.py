
## TO RUN FROM THE TF-SLIM/MODELS/SLIM DIRECTORY

import datasets.dataset_utils as dataset_utils
import datasets.imagenet as imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.framework import graph_util
import os

url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

# Specify where you want to download the model to
checkpoints_dir = '/media/sf_tensorflow_mount/checkpoints'
image_path = '/media/sf_tensorflow_mount/ant.jpg'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

#dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

slim = tf.contrib.slim

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = vgg.vgg_16.default_image_size

def get_op_name(tensor):
    return tensor.name.split(":")[0]

# Build the graph
g = tf.Graph()
with g.as_default():
    # Open specified url and load image as a string
    image_string = open(image_path, 'rb').read()

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

    output_nodes = [probabilities, top_pred.indices, top_pred.values]


# Create the saver
with g.as_default():

    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    # init_fn = slim.assign_from_checkpoint_fn(
    #     os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
    #     slim.get_model_variables('vgg_16'))

    checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')
    model_variables = slim.get_model_variables('vgg_16')
    saver = tf_saver.Saver(model_variables, reshape=False)

# # Test the initial network
# with g.as_default():
#     with tf.Session() as sess:
#         saver.restore(sess, checkpoint_path)
#
#         # Load weights
#         #init_fn(sess)
#
#         # We want to get predictions, image as numpy matrix
#         # and resized and cropped piece that is actually
#         # being fed to the network.
#         output_nodes = [probabilities, top_pred.indices, top_pred.values]
#         probabilities_, indices_, values_ = sess.run(output_nodes)
#         probabilities_ = probabilities_[0, 0:]
#         sorted_inds = [i[0] for i in sorted(enumerate(-probabilities_),
#                                             key=lambda x:x[1])]

# Export the network
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

del g

g2 = tf.Graph()
with g2.as_default():
    tf.import_graph_def(output_graph_def, name='')

del output_graph_def

# # Test the exported network
# image_data = tf.gfile.FastGFile(image_path, 'rb').read()
# with g2.as_default():
#     input_node2 = g2.get_operation_by_name(get_op_name(image))
#     output_nodes2 = [g2.get_tensor_by_name(n) for n in output_tensor_names]
#     with tf.Session() as sess:
#         (probabilities_, indices_, values_) = sess.run(output_nodes2, {'DecodeJpeg/contents:0':image_data})
#
# names = imagenet.create_readable_names_for_imagenet_labels()
# for i in range(5):
#     index = indices_[i]
#     # Now we print the top-5 predictions that the network gives us with
#     # corresponding probabilities. Pay attention that the index with
#     # class names is shifted by 1 -- this is because some networks
#     # were trained on 1000 classes and others on 1001. VGG-16 was trained
#     # on 1000 classes.
#     print('Probability %d %0.2f => [%s]' % (index, values_[i], names[index+1]))


## Using Spark

import tensorframes as tfs
sc.setLogLevel('INFO')

with g2.as_default():
    index_output = tf.identity(g2.get_tensor_by_name('top_predictions:1'), name="index")
    value_output = tf.identity(g2.get_tensor_by_name('top_predictions:0'), name="value")


raw_images_miscast = sc.binaryFiles("file:"+image_path) # file:
raw_images = raw_images_miscast.map(lambda x: (x[0], bytearray(x[1])))

df = spark.createDataFrame(raw_images).toDF('image_uri', 'image_data')
df


with g2.as_default():
    pred_df = tfs.map_rows([index_output, value_output], df, feed_dict={'DecodeJpeg/contents':'image_data'})

pred_df.select('index', 'value').head()



## Tim
#
# raw_images_miscast = sc.binaryFiles("file:/media/sf_tensorflow_mount/101_ObjectCategories/ant/")
# raw_images = raw_images_miscast.map(lambda x: (x[0], bytearray(x[1])))
#
# df = spark.createDataFrame(raw_images).toDF('image_uri', 'image_data')
# df
# with g2.as_default():
#     pred_df = tfs.map_rows([index_output, value_output], df, feed_dict={'DecodeJpeg/contents':'image_data'})
#
# pred_df.select('index', 'value').show()

