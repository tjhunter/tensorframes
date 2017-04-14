"""
Implementation of the K-Means algorithm, while distributing the computations on a cluster.

Given a set of feature vectors, this algorithm runs the K-Means clustering algorithm starting
from a given set of centroids.
"""
from __future__ import print_function

import tensorflow as tf
import tensorframes as tfs
import numpy as np

def tf_compute_distances(points, start_centers):
    """
    Given a set of points and some centroids, computes the distance from each point to each
    centroid.

    :param points: a 2d TF tensor of shape num_points x dim
    :param start_centers: a numpy array of shape num_centroid x dim
    :return: a TF tensor of shape num_points x num_centroids
    """
    with tf.variable_scope("distances"):
        # The dimensions in the problem
        (num_centroids, _) = np.shape(start_centers)
        # The shape of the block is extracted as a TF variable.
        num_points = tf.shape(points)[0]
        # The centers are embedded in the TF program.
        centers = tf.constant(start_centers)
        # Computation of the minimum distance. This is a standard implementation that follows
        # what MLlib does.
        squares = tf.reduce_sum(tf.square(points), reduction_indices=1)
        center_squares = tf.reduce_sum(tf.square(centers), reduction_indices=1)
        prods = tf.matmul(points, centers, transpose_b = True)
        # This code simply expresses two outer products: center_squares * ones(num_points)
        # and ones(num_centroids) * squares
        t1a = tf.expand_dims(center_squares, 0)
        t1b = tf.stack([num_points, 1])
        t1 = tf.tile(t1a, t1b)
        t2a = tf.expand_dims(squares, 1)
        t2b = tf.stack([1, num_centroids])
        t2 = tf.tile(t2a, t2b)
        distances = t1 + t2 - 2 * prods
    return distances


def run_one_step(dataframe, start_centers):
    """
    Performs one iteration of K-Means.

    This function takes a dataframe with dense feature vectors, a set of centroids, and returns
    a new set of centroids along with the total distance of points to centroids.

    This function calculates for each point the closest centroid and then aggregates the newly
    formed clusters to find the new centroids.

    This function uses Spark to distribute the aggregation amongst the node.

    :param dataframe: a dataframe containing a column of features (an array of doubles)
    :param start_centers: a k x m matrix with k the number of centroids and m the number of features
    :return: a k x m matrix, and a positive double
    """
    # The dimensions in the problem
    (num_centroids, num_features) = np.shape(start_centers)
    # For each feature vector, compute the nearest centroid and the distance to that centroid.
    # The index of the nearest centroid is stored in the 'indexes' column.
    # We also add a column of 1's that will be reduced later to count the number of elements in
    # each cluster.
    with tf.Graph().as_default() as g:
        # The placeholder for the input: we use the block format
        points = tf.placeholder(tf.double, shape=[None, num_features], name='features')
        # The shape of the block is extracted as a TF variable.
        num_points = tf.stack([tf.shape(points)[0]], name="num_points")
        distances = tf_compute_distances(points, start_centers)
        # The outputs of the program.
        # The closest centroids are extracted.
        indexes = tf.argmin(distances, 1, name='indexes')
        # This could be done based on the indexes as well.
        min_distances = tf.reduce_min(distances, 1, name='min_distances')
        counts = tf.tile(tf.constant([1]), num_points, name='count')
        df2 = tfs.map_blocks([indexes, counts, min_distances], dataframe)
    # Perform the reduction: we regroup the points by their centroid indexes.
    gb = df2.groupBy("indexes")
    with tf.Graph().as_default() as g:
        # Look at the documentation of tfs.aggregate for the naming conventions of the placeholders.
        x_input = tfs.block(df2, "features", tf_name="features_input")
        count_input = tfs.block(df2, "count", tf_name="count_input")
        md_input = tfs.block(df2, "min_distances", tf_name="min_distances_input")
        # Each operation is just the sum.
        x = tf.reduce_sum(x_input, [0], name='features')
        count = tf.reduce_sum(count_input, [0], name='count')
        min_distances = tf.reduce_sum(md_input, [0], name='min_distances')
        df3 = tfs.aggregate([x, count, min_distances], gb)
    # Get the new centroids
    df3_c = df3.collect()
    # The new centroids.
    new_centers = np.array([np.array(row.features) / row['count'] for row in df3_c])
    total_distances = np.sum([row['min_distances'] for row in df3_c])
    return (new_centers, total_distances)


def run_one_step2(dataframe, start_centers):
    """
    Performs one iteration of K-Means.

    This function takes a dataframe with dense feature vectors, a set of centroids, and returns
    a new set of centroids along with the total distance of points to centroids.

    This function calculates for each point the closest centroid and then aggregates the newly
    formed clusters to find the new centroids.

    This function performs most of the aggregation in TensorFlow.

    :param dataframe: a dataframe containing a column of features (an array of doubles)
    :param start_centers: a k x m matrix with k the number of centroids and m the number of features
    :return: a k x m matrix, and a positive double
    """
    # The dimensions in the problem
    (num_centroids, _) = np.shape(start_centers)
    # For each feature vector, compute the nearest centroid and the distance to that centroid.
    # The index of the nearest centroid is stored in the 'indexes' column.
    # We also add a column of 1's that will be reduced later to count the number of elements in
    # each cluster.
    with tf.Graph().as_default() as g:
        # The placeholder for the input: we use the block format
        points = tf.placeholder(tf.double, shape=[None, num_features], name='features')
        # The distances
        distances = tf_compute_distances(points, start_centers)
        # The rest of this block performs a pre-aggregation step in TF, to limit the
        # communication between TF and Spark.
        # The closest centroids are extracted.
        indexes = tf.argmin(distances, 1, name='indexes')
        min_distances = tf.reduce_min(distances, 1, name='min_distances')
        num_points = tf.stack([tf.shape(points)[0]], name="num_points")
        counts = tf.tile(tf.constant([1]), num_points, name='count')
        # These compute the aggregate based on the indexes.
        block_points = tf.unsorted_segment_sum(points, indexes, num_centroids, name="block_points")
        block_counts = tf.unsorted_segment_sum(counts, indexes, num_centroids, name="block_counts")
        block_distances = tf.reduce_sum(min_distances, name="block_distances")
        # One leading dimension is added to express the fact that the previous elements are just
        # one row in the final dataframe.
        # The final dataframe has one row per block.
        agg_points = tf.expand_dims(block_points, 0, name="agg_points")
        agg_counts = tf.expand_dims(block_counts, 0, name="agg_counts")
        agg_distances = tf.expand_dims(block_distances, 0, name="agg_distances")
        # Using trimming to drop the original data (we are just returning one row of data per
        # block).
        df2 = tfs.map_blocks([agg_points, agg_counts, agg_distances],
                             dataframe, trim=True)
    # Now we simply collect and sum the elements
    with tf.Graph().as_default() as g:
        # Look at the documentation of tfs.aggregate for the naming conventions of the placeholders.
        x_input = tf.placeholder(tf.double,
                                 shape=[None, num_centroids, num_features],
                                 name='agg_points_input')
        count_input = tf.placeholder(tf.int32,
                                     shape=[None, num_centroids],
                                     name='agg_counts_input')
        md_input = tf.placeholder(tf.double,
                                  shape=[None],
                                  name='agg_distances_input')
        # Each operation is just the sum.
        x = tf.reduce_sum(x_input, [0], name='agg_points')
        count = tf.reduce_sum(count_input, [0], name='agg_counts')
        min_distances = tf.reduce_sum(md_input, [0], name='agg_distances')
        (x_, count_, total_distances) = tfs.reduce_blocks([x, count, min_distances], df2)
    # The new centers
    new_centers = (x_.T / (count_ + 1e-7)).T
    return (new_centers, total_distances)


def kmeanstf(dataframe, init_centers, num_iters = 5, tf_aggregate = True):
    """
    Runs the K-Means algorithm on a set of feature points.

    This function takes a dataframe with dense feature vectors, a set of centroids, and returns
    a new set of centroids along with the total distance of points to centroids.

    :param dataframe: a dataframe containing a column of features (an array of doubles)
    :param init_centers: the centers to start from
    :param num_iters:  the maximum number of iterations to run
    :return: a k x m matrix, and a list of positive doubles
    """
    step_fun = run_one_step2 if tf_aggregate else run_one_step
    c = init_centers
    d = np.Inf
    ds = []
    for i in range(num_iters):
        (c1, d1) = step_fun(dataframe, c)
        print("Step =", i, ", overall distance = ", d1)
        c = c1
        if d == d1:
            break
        d = d1
        ds.append(d1)
    return c, ds

# Here is a an example of usage:
try:
    sc.setLogLevel('INFO')
except:
    pass

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.linalg import VectorUDT, _convert_to_vector
from pyspark.sql.types import Row, StructField, StructType
import time

# Small vectors
num_features = 100
# The number of clusters
k = 10
num_points = 100000
num_iters = 10
FEATURES_COL = "features"

np.random.seed(2)
np_data = [x.tolist() for x in np.random.uniform(0.0, 1.0, size=(num_points, num_features))]
schema = StructType([StructField(FEATURES_COL, VectorUDT(), False)])
mllib_rows = [Row(_convert_to_vector(x)) for x in np_data]
mllib_df = sqlContext.createDataFrame(mllib_rows, schema).coalesce(1).cache()

df = sqlContext.createDataFrame([[r] for r in np_data]).toDF(FEATURES_COL).coalesce(1)
# For now, analysis is still required. We cache the output because we are going to perform
# multiple runs on the dataset.
df0 = tfs.analyze(df).cache()


mllib_df.count()
df0.count()

np.random.seed(2)
init_centers = np.random.randn(k, num_features)
start_centers = init_centers
dataframe = df0


ta_0 = time.time()
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol(FEATURES_COL).setInitMode(
        "random").setMaxIter(num_iters)
mod = kmeans.fit(mllib_df)
ta_1 = time.time()

tb_0 = time.time()
(centers, agg_distances) = kmeanstf(df0, init_centers, num_iters=num_iters, tf_aggregate=False)
tb_1 = time.time()

tc_0 = time.time()
(centers, agg_distances) = kmeanstf(df0, init_centers, num_iters=num_iters, tf_aggregate=True)
tc_1 = time.time()

mllib_dt = ta_1 - ta_0
tf_dt = tb_1 - tb_0
tf2_dt = tc_1 - tc_0

print("mllib:", mllib_dt, "tf+spark:",tf_dt, "tf:",tf2_dt)
