
from __future__ import print_function

from pyspark import SparkContext
from pyspark.sql import DataFrame, SQLContext
from pyspark.sql import Row
import tensorflow as tf
import pandas as pd

import tensorframes as tfs
from tensorframes.core import _java_api

class TestCore(object):

    @classmethod
    def setup_class(cls):
        print("setup ", cls)
        cls.sc = SparkContext('local[1]', cls.__name__)

    @classmethod
    def teardown_class(cls):
        print("teardown ", cls)
        cls.sc.stop()

    def setUp(self):
        self.sql = SQLContext(TestCore.sc)
        self.api = _java_api()
        self.api.initialize_logging()
        print("setup")


    def teardown(self):
        print("teardown")

    def test_schema(self):
        data = [Row(x=float(x)) for x in range(100)]
        df = self.sql.createDataFrame(data)
        tfs.print_schema(df)

    def test_map_blocks_0(self):
        data = [Row(x=float(x)) for x in range(10)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            y = tf.Variable(3.0, dtype=tf.double, name='y')
            z = tf.add(x, y, name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df)
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_blocks_1(self):
        data = [Row(x=float(x)) for x in range(10)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df)
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_blocks_2(self):
        data = [dict(x=float(x)) for x in range(10)]
        df = pd.DataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df)
        data2 = df2
        assert data2.z[0] == 3.0, data2

    def test_map_blocks_3(self):
        data = [dict(x=float(x)) for x in range(10)]
        df = pd.DataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            y = tf.Variable(3.0, dtype=tf.double, name='y')
            z = tf.add(x, y, name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df)
        data2 = df2
        assert data2.z[0] == 3.0, data2

    def test_map_blocks_feed_dict(self):
        data = [dict(x_spark=float(x)) for x in range(10)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x_tf")
            # The output that adds 3 to x
            y = tf.Variable(3.0, dtype=tf.double, name='y')
            z = tf.add(x, y, name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df, feed_dict={'x_tf': 'x_spark'})
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_rows_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_rows(z, df)
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_rows_2(self):
        data = [Row(y=float(y)) for y in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_rows(z, df, feed_dict={'x':'y'})
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_rows_3(self):
        data = [dict(x=float(x)) for x in range(5)]
        df = pd.DataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_rows(z, df)
        data2 = df2
        assert data2.z[0] == 3.0, data2

    def test_map_rows_feed_dict(self):
        data = [dict(x_spark=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[], name="x_tf")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_rows(z, df, feed_dict={'x_tf': 'x_spark'})
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_rows_4(self):
        data = [dict(y=float(x)) for x in range(5)]
        df = pd.DataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_rows(z, df, feed_dict={'x':'y'})
        data2 = df2
        assert data2.z[0] == 3.0, data2

    def test_reduce_rows_0(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x_1 = tf.placeholder(tf.double, shape=[], name="x_1")
            x_2 = tf.placeholder(tf.double, shape=[], name="x_2")
            y = tf.Variable(0.0, dtype=tf.double, name='y')
            x_0 = tf.add(y, x_1, name='x_0')
            # The output that adds 3 to x
            x = tf.add(x_0, x_2, name='x')
            # The resulting number
            res = tfs.reduce_rows(x, df)
        assert res == sum([r.x for r in data])

    def test_reduce_rows_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x_1 = tf.placeholder(tf.double, shape=[], name="x_1")
            x_2 = tf.placeholder(tf.double, shape=[], name="x_2")
            # The output that adds 3 to x
            x = tf.add(x_1, x_2, name='x')
            # The resulting number
            res = tfs.reduce_rows(x, df)
        assert res == sum([r.x for r in data])

    # This test fails
    def test_reduce_blocks_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x_input = tf.placeholder(tf.double, shape=[None], name="x_input")
            # The output that adds 3 to x
            x = tf.reduce_sum(x_input, name='x')
            # The resulting dataframe
            res = tfs.reduce_blocks(x, df)
        assert res == sum([r.x for r in data])

    def test_map_blocks_trimmed_0(self):
        data = [Row(x=float(x)) for x in range(3)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output discards the input and return a single row of data
            z = tf.Variable([2], dtype=tf.double, name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df, trim=True)
        data2 = df2.collect()
        assert data2[0].z == 2, data2

    def test_map_blocks_trimmed_1(self):
        data = [Row(x=float(x)) for x in range(3)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output discards the input and return a single row of data
            z = tf.constant([2], name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df, trim=True)
        data2 = df2.collect()
        assert data2[0].z == 2, data2

    def test_groupby_1(self):
        data = [Row(x=float(x), key=str(x % 2)) for x in range(4)]
        df = self.sql.createDataFrame(data)
        gb = df.groupBy("key")
        with tf.Graph().as_default():
            x_input = tfs.block(df, "x", tf_name="x_input")
            x = tf.reduce_sum(x_input, [0], name='x')
            df2 = tfs.aggregate(x, gb)
        data2 = df2.collect()
        assert data2 == [Row(key='0', x=2.0), Row(key='1', x=4.0)], data2


if __name__ == "__main__":
    # Some testing stuff that should not be executed
    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.double, shape=[2, 3], name="x_input")
        x = tf.reduce_sum(x_input, [0], name='x')
        print(g.as_graph_def())

    with tf.Graph().as_default():
        x = tf.constant([1, 1], name="x")
        y = tf.reduce_sum(x, [0], name='y')
        print(g.as_graph_def())

    with tf.Graph().as_default():
        tf.constant(1, name="x1")
        tf.constant(1.0, name="x2")
        tf.constant([1.0], name="x3")
        tf.constant([1.0, 2.0], name="x4")
        print(g.as_graph_def())
