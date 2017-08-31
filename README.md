![build](https://travis-ci.org/databricks/tensorframes.svg)

# TensorFrames

Experimental [TensorFlow](https://www.tensorflow.org/) binding for Scala and 
[Apache Spark](http://spark.apache.org/).

TensorFrames (TensorFlow on Spark Dataframes) lets you manipulate Apache Spark's DataFrames with 
TensorFlow programs.

> This package is experimental and is provided as a technical preview only. While the 
> interfaces are all implemented and working, there are still some areas of low performance.

Supported platforms:

> This package only officially supports linux 64bit platforms as a target.
> Contributions are welcome for other platforms.

See the file `project/Dependencies.scala` for adding your own platform.

Officially supported Spark versions: 2.1.x and Scala version 2.10 / 2.11.

See the [user guide](https://github.com/databricks/tensorframes/wiki/TensorFrames-user-guide) for
 extensive information about the API.

For questions, see the [TensorFrames mailing list](https://groups.google.com/forum/#!forum/tensorframes).

TensorFrames is available as a
 [Spark package](http://spark-packages.org/package/databricks/tensorframes).

## Requirements

 - A working version of Apache Spark (2.0 or greater)

 - java version >= 7
 
 - (Optional) python >= 2.7, or python >= 3.4 if you want to use the python interface.
 
 - (Optional) the python TensorFlow package if you want to use the python interface. See the 
 [official instructions](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#download-and-setup)
  on how to get the latest release of TensorFlow.

 - (Optional) pandas >= 0.19.1 if you want to use the python interface

 - (Optional) the [Nix package manager](http://nixos.org/nix/) if you want to guarantee a fully reproducible build environment. This is the environment that will be used for reproducing bugs.

Additionally, if you want to run unit tests for python, you need the following dependencies:

 - nose >= 1.3 


## How to run in python

Assuming that `SPARK_HOME` is set, you can use PySpark like any other Spark package.

```bash
$SPARK_HOME/bin/pyspark --packages databricks:tensorframes:0.2.9-rc3-s_2.11
```

Here is a small program that uses Tensorflow to add 3 to an existing column.

```python
import tensorflow as tf
import tensorframes as tfs
from pyspark.sql import Row

data = [Row(x=float(x)) for x in range(10)]
df = sqlContext.createDataFrame(data)
with tf.Graph().as_default() as g:
    # The TensorFlow placeholder that corresponds to column 'x'.
    # The shape of the placeholder is automatically inferred from the DataFrame.
    x = tfs.block(df, "x")
    # The output that adds 3 to x
    z = tf.add(x, 3, name='z')
    # The resulting dataframe
    df2 = tfs.map_blocks(z, df)

# The transform is lazy as for most DataFrame operations. This will trigger it:
df2.collect()

# Notice that z is an extra column next to x

# [Row(z=3.0, x=0.0),
#  Row(z=4.0, x=1.0),
#  Row(z=5.0, x=2.0),
#  Row(z=6.0, x=3.0),
#  Row(z=7.0, x=4.0),
#  Row(z=8.0, x=5.0),
#  Row(z=9.0, x=6.0),
#  Row(z=10.0, x=7.0),
#  Row(z=11.0, x=8.0),
#  Row(z=12.0, x=9.0)]
```

The second example shows the block-wise reducing operations: we compute the sum of a field containing 
vectors of integers, working with blocks of rows for more efficient processing.

```python
# Build a DataFrame of vectors
data = [Row(y=[float(y), float(-y)]) for y in range(10)]
df = sqlContext.createDataFrame(data)
# Because the dataframe contains vectors, we need to analyze it first to find the
# dimensions of the vectors.
df2 = tfs.analyze(df)

# The information gathered by TF can be printed to check the content:
tfs.print_schema(df2)
# TF has inferred that y contains vectors of size 2
# root
#  |-- y: array (nullable = false) DoubleType[?,2]

# Let's use the analyzed dataframe to compute the sum and the elementwise minimum 
# of all the vectors:
# First, let's make a copy of the 'y' column. This will be very cheap in Spark 2.0+
df3 = df2.select(df2.y, df2.y.alias("z"))
with tf.Graph().as_default() as g:
    # The placeholders. Note the special name that end with '_input':
    y_input = tfs.block(df3, 'y', tf_name="y_input")
    z_input = tfs.block(df3, 'z', tf_name="z_input")
    y = tf.reduce_sum(y_input, [0], name='y')
    z = tf.reduce_min(z_input, [0], name='z')
    # The resulting dataframe
    (data_sum, data_min) = tfs.reduce_blocks([y, z], df3)

# The final results are numpy arrays:
print data_sum
# [45.0, -45.0]
print data_min
# [0.0, -9.0]
```

*Notes*

Note the scoping of the graphs above. This is important because TensorFrames finds which 
DataFrame column to feed to TensorFrames based on the placeholders of the graph. Also, it is 
 good practice to keep small graphs when sending them to Spark.
 
For small tensors (scalars and vectors), TensorFrames usually infers the shapes of the 
tensors without requiring a preliminary analysis. If it cannot do it, an error message will 
indicate that you need to run the DataFrame through `tfs.analyze()` first.

Look at the python documentation of the TensorFrames package to see what methods are available.


## How to run in Scala

The scala support is a bit more limited than python. In scala, operations can be loaded from 
 an existing graph defined in the ProtocolBuffers format, or using a simple scala DSL. The
 Scala DSL only features a subset of TensorFlow transforms. It is very easy to extend
 though, so other transforms will be added without much effort in the future.

You simply use the published package:

```bash
$SPARK_HOME/bin/spark-shell --packages databricks:tensorframes:0.2.9-rc3
```

Here is the same program as before:

```scala
import org.tensorframes.{dsl => tf}
import org.tensorframes.dsl.Implicits._

val df = spark.createDataFrame(Seq(1.0->1.1, 2.0->2.2)).toDF("a", "b")

// As in Python, scoping is recommended to prevent name collisions.
val df2 = tf.withGraph {
    val a = df.block("a")
    // Unlike python, the scala syntax is more flexible:
    val out = a + 3.0 named "out"
    // The 'mapBlocks' method is added using implicits to dataframes.
    df.mapBlocks(out).select("a", "out")
}

// The transform is all lazy at this point, let's execute it with collect:
df2.collect()
// res0: Array[org.apache.spark.sql.Row] = Array([1.0,1.1,2.1], [2.0,2.2,4.2])   
```

## How to compile and install for developers

It is recommended you use [Nix](http://nixos.org/nix/) to guarantee that the build environment
can be reproduced. Once you have installed Nix, you can set the environment from
the root of project:

```bash
nix-shell --pure default.nix
```

This will create a python 2.7 environment with all the dependencies. If you
want to work with Python 3.5, use `default-3.5.nix` instead.

 The C++ bindings are already compiled though, so you should only have to deal with compiling
 the scala code. The recommended procedure is to use the assembly:

```bash
build/sbt tfs_testing/assembly
# Builds the spark package:
build/sbt distribution/spDist
```

Assuming that SPARK_HOME is set and that you are in the root directory of the project:

```bash
$SPARK_HOME/bin/spark-shell --jars $PWD/target/testing/scala-2.11/tensorframes-assembly-0.2.9-rc3.jar
```

If you want to run the python version:
 
```bash
PYTHONPATH=$PWD/target/testing/scala-2.11/tensorframes-assembly-0.2.9-rc3.jar \
$SPARK_HOME/bin/pyspark --jars $PWD/target/testing/scala-2.11/tensorframes-assembly-0.2.9-rc3.jar
```

## Acknowledgements

This project builds on the great [javacpp](https://github.com/bytedeco/javacpp) project, that
 implements the low-level bindings between TensorFlow and the Java virtual machine.

Many thanks to Google for the release of TensorFlow.
