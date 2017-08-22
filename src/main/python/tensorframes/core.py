import tensorflow as tf
import numpy as np
import logging
import tempfile
import pandas as pd

from pyspark import RDD, SparkContext
from pyspark.sql import SQLContext, Row, DataFrame
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType, ArrayType

__all__ = ['reduce_rows', 'map_rows', 'reduce_blocks', 'map_blocks',
           'analyze', 'print_schema', 'aggregate', 'block', 'row']

_sc = None
_sql = None
_initial_variables_default = True
logger = logging.getLogger('tensorframes')
first_tensor_by_op_name = lambda graph, name: graph.get_operation_by_name(name).outputs[0]

def _java_api():
    """
    Loads the PythonInterface object (lazily, because the spark context needs to be initialized
    first).
    """
    global _sc, _sql
    javaClassName = "org.tensorframes.impl.DebugRowOps"
    if _sc is None:
        _sc = SparkContext._active_spark_context
        logger.info("Spark context = " + str(_sc))
        _sql = SQLContext(_sc)
    _jvm = _sc._jvm
    # You cannot simply call the creation of the the class on the _jvm due to classloader issues
    # with Py4J.
    return _jvm.Thread.currentThread().getContextClassLoader().loadClass(javaClassName) \
        .newInstance()

def _get_shape(node):
    l = node.get_shape().as_list()
    return [-1 if x is None else x for x in l]

def _initialize_variables(graph, fetches, initial_variables):
    with tf.Session(graph=graph) as sess:
        if len(tf.global_variables()) == 0:
            return graph, fetches

        if initial_variables:
            sess.run(tf.global_variables_initializer())
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            graph.as_graph_def(add_shapes=True),
                            [fetch.op.name for fetch in fetches])

    new_graph = tf.Graph()
    with new_graph.as_default():
        tf.import_graph_def(output_graph_def, name='')
        return new_graph, [new_graph.get_tensor_by_name(fetch.name) for fetch in fetches]

def _add_graph(graph, builder, use_file=True):
    if use_file:
        # TODO: remove the dir and honor the existing one
        d = tempfile.mkdtemp("tensorframes", dir="/tmp")
        tf.train.write_graph(graph, d, "proto.pb", False)
        fname = d + "/proto.pb"
        builder.graphFromFile(fname)
    else:
        # Make sure that TF adds the shapes.
        gser = graph.as_graph_def(add_shapes=True).SerializeToString()
        gbytes = bytearray(gser)
        builder.graph(gbytes)

# Returns the names of the placeholders.
def _add_shapes(graph, builder, fetches):
    names = [fetch.name for fetch in fetches]
    shapes = [_get_shape(fetch) for fetch in fetches]
    # We still need to do the placeholders, it seems their shape is not passed in when some
    # dimensions are unknown
    ph_names = []
    ph_shapes = []
    for n in graph.as_graph_def(add_shapes=True).node:
        # Just the input nodes:
        if not n.input:
            op_name = n.name
            # Simply get the default output for now, assume that the nodes have only one output
            t = graph.get_tensor_by_name(op_name + ":0")
            ph_names.append(t.name)
            ph_shapes.append(_get_shape(t))
    logger.info("fetches: %s %s", str(names), str(shapes))
    logger.info("inputs: %s %s", str(ph_names), str(ph_shapes))
    builder.shape(names + ph_names, shapes + ph_shapes)
    builder.fetches(names)
    # return the path, not the tensor name.
    return [t_name.replace(":0", "") for t_name in ph_names]

def _check_fetches(fetches):
    is_list_fetch = isinstance(fetches, (list, tuple))
    if not is_list_fetch:
        return [fetches]
    return fetches

def _get_graph(fetches, initial_variables=_initial_variables_default):
    graph = tf.get_default_graph()
    graph, fetches = _initialize_variables(graph, fetches, initial_variables)
    fetch_names = [_validate_fetch(graph, fetch) for fetch in fetches]
    logger.info("Fetch names: %s", str(fetch_names))
    # String the output index
    col_names = [s.split(":")[0] for s in fetch_names]
    if len(set(col_names)) != len(col_names):
        raise ValueError("Could not infer a list of unique names for the columns: %s" % str(fetch_names))
    return graph

def _unpack_row(jdf, fetches):
    df = DataFrame(jdf, _sql)
    row = df.first()
    def f(fetch):
        name = fetch.name.replace(":0", "")
        x = row[name]
        ndims = fetch.get_shape().ndims
        if ndims > 0:
            return np.array(x)
        else:
            return x
    l = [f(fetch) for fetch in fetches]
    if len(l) == 1:
        return l[0]
    return l


def _add_inputs(builder, start_dct, ph_names):
    """
    Combines a dictionary (supplied by the user) with some extra placeholder names.
    """
    if start_dct is None:
        start_dct = {}
    dct = dict(**start_dct)
    for ph_name in ph_names:
        if ph_name not in dct:
            dct[ph_name] = ph_name
    dct_items = dct.items()
    input_names = [ph_name for (ph_name, field_name) in dct_items]
    field_names = [field_name for (ph_name, field_name) in dct_items]
    logger.info("inputs: %s %s", str(input_names), str(field_names))
    builder.inputs(input_names, field_names)

def _map(fetches, dframe, feed_dict, block, trim, initial_variables=_initial_variables_default):
    fetches = _check_fetches(fetches)
    # We are not dealing for now with registered expansions, but this is something we should add later.
    graph = _get_graph(fetches, initial_variables)
    if block:
        builder = _java_api().map_blocks(dframe._jdf, trim)
    else:
        builder = _java_api().map_rows(dframe._jdf)
    _add_graph(graph, builder)
    ph_names = _add_shapes(graph, builder, fetches)
    _add_inputs(builder, feed_dict, ph_names)
    jdf = builder.buildDF()
    return DataFrame(jdf, _sql)

def _get_input(graph):
    ph_names = []
    for n in graph.as_graph_def().node:
        # Just the input nodes:
        if n.op == 'Placeholder':
            ph_names.append(n.name)
    return ph_names

def _unique_graph(fetches):
    graph = set([t.graph for t in fetches])
    assert len(graph)==1, 'there can only be one graph'
    return graph.pop()

def _map_pd(fetches, pdframe, feed_dict=None, block=None, trim=None, initial_variables=_initial_variables_default):
    fetches = _check_fetches(fetches)
    graph = _unique_graph(fetches)
    graph, fetches = _initialize_variables(graph, fetches, initial_variables)
    if feed_dict is None:
        feed_names = _get_input(graph)
        feed_dict = dict(zip(feed_names, feed_names))

    with tf.Session(graph=graph) as sess:
        res = sess.run(fetches, feed_dict={first_tensor_by_op_name(graph, k): pdframe[feed_dict[k]] for k in feed_dict})
        for v, n in zip(fetches, res):
            pdframe[v.op.name] = n
        return pdframe

def reduce_rows(fetches, dframe, initial_variables=_initial_variables_default):
    """ Applies the fetches on pairs of rows, so that only one row of data remains in the end. The order in which
    the operations are performed on the rows is unspecified.

    The `fetches` argument may be a list of graph elements or a single
    graph element. A graph element can be of the following type:

    * If the *i*th element of `fetches` is a
      `Tensor`, the *i*th return value will be a numpy ndarray containing the value of that tensor.

    There is no support for sparse tensor objects yet.

    This transform not lazy and is performed when called.

    In order to perform the reduce operation, the fetches must follow some naming conventions: for each fetch called
    for example 'z', there must be two placeholders 'z_1' and 'z_2' that will be fed with the input data. The shapes
    and the dtypes of z, z_1 and z_2 must be the same.

    Args:
      fetches: A single graph element, or a list of graph elements
        (described above).
      dframe: A DataFrame object. The columns of the tensor frame will be fed into the fetches at execution.
      initial_variables: a boolean option default True, initial variables if it is used.

    Returns: a list of numpy arrays, one for each of the fetches, or a single numpy array if there is but one fetch.

    :param fetches: see description above
    :param dframe: a Spark DataFrame
    :param initial_variables: a boolean option default True, initial variables if it is used.
    :return: a list of numpy arrays
    """
    fetches = _check_fetches(fetches)
    graph = _get_graph(fetches, initial_variables)
    builder = _java_api().reduce_rows(dframe._jdf)
    _add_graph(graph, builder)
    _add_shapes(graph, builder, fetches)
    df = builder.buildRow()
    return _unpack_row(df, fetches)

def map_rows(fetches, dframe, feed_dict=None, initial_variables=_initial_variables_default):
    """ Transforms a DataFrame into another DataFrame row by row, by adding new fields for each fetch.

    The `fetches` argument may be a list of graph elements or a single
    graph element. A graph element can be one of the following type:

    * a TensorFlow's Tensor object. The shape and the dtype of the tensor will dictate the structure of the column

    Note on computations: unlike the TensorFlow execution engine, the result is lazy and will not be computed until
    requested. However, all the fetches and the computation graph are frozen when this function is called.

    The inputs of the fetches must all be constants or placeholders. The placeholders must have the name of existing
    fields in the dataframe, and they must have the same dtype as the placeholder (no implicit casting is performed on
    the input). Additionally, they must have a tensor shape that is compatible with the shape of the elements in that
    field. For example, if the field contains scalar, only scalar shapes are accepted, etc.

    The names of the fetches must be all different from the names of existing columns, otherwise an error is returned.

    This method works row by row. If you want a more efficient method that can work on batches of rows, consider using
    [map_blocks] instead.

    Args:
      fetches: A single graph element, or a list of graph elements
        (described above).
      dframe: A Spark DataFrame object or a pandas DataFrame(for debug purpose).
        The columns of the tensor frame will be fed into the fetches at execution.
      feed_dict: a dictionary of string -> string. The key is the name of a placeholder in the current TensorFlow graph
                 of computation. The value is the name of a column in the dataframe. For now, only the top-level fields
                 in a dataframe are supported. For any placeholder that is not specified in the feed dictionary, the
                 name of the input column is assumed to be the same as that of the placeholder.
      initial_variables: a boolean option default True, initial variables if it is used.

    Returns: a DataFrame. The columns and their names are inferred from the names of the fetches.

    :param fetches: see description above
    :param dframe: a Spark DataFrame or a pandas DataFrame
    :param initial_variables: a boolean option default True, initial variables if it is used.
    :return: a Spark DataFrame or a pandas DataFrame
    """
    if isinstance(dframe, pd.DataFrame):
        return _map_pd(fetches, dframe, feed_dict, block=False, trim=None, initial_variables=initial_variables)
    return _map(fetches, dframe, feed_dict, block=False, trim=None, initial_variables=initial_variables)

def map_blocks(fetches, dframe, feed_dict=None, trim=False, initial_variables=_initial_variables_default):
    """ Transforms a DataFrame into another DataFrame block by block.

    It either appends new columns to the DataFrame (trim = false), or it completely discards the
    inputs and only returns the new fields produced by TensorFlow (trim = true).

    The `fetches` argument may be a list of graph elements or a single
    graph element. A graph element can be one of the following type:

    * a TensorFlow's Tensor object. The shape and the dtype of the tensor will dictate the structure of the column

    Note on computations: unlike the TensorFlow execution engine, the result is lazy and will not be computed until
    requested. However, all the fetches and the computation graph are frozen when this function is called.

    The inputs of the fetches must all be constants or placeholders. The placeholders must have the name of existing
    fields in the dataframe, and they must have the same dtype as the placeholder (no implicit casting is performed on
    the input). Additionally, they must have a tensor shape that is compatible with the shape of the elements in that
    field. For example, if the field contains scalar, only scalar shapes are accepted, etc.

    The names of the fetches must be all different from the names of existing columns, otherwise an error is returned.

    This method does not work when rows contains vectors of different sizes. In this case,
    you must use [map_rows].

    If trim is true (the output only contains fields generated by tensorflow), then the number of
    rows being produced is allowed to differ from the number of rows of the input block.

    Args:
      fetches: A single graph element, or a list of graph elements
        (described above).
      dframe: A Spark DataFrame object or a pandas DataFrame(for debug purpose).
        The columns of the tensor frame will be fed into the fetches at execution.
      feed_dict: a dictionary of string -> string. The key is the name of a placeholder in the current TensorFlow graph
                 of computation. The value is the name of a column in the dataframe. For now, only the top-level fields
                 in a dataframe are supported. For any placeholder that is not specified in the feed dictionary, the
                 name of the input column is assumed to be the same as that of the placeholder.
      trim: if true, only the fields generated by the TensorFlow graph will be returned.

    Returns: a DataFrame. The columns and their names are inferred from the names of the fetches.

    :param fetches: see description above
    :param dframe: a Spark DataFrame or a pandas DataFrame
    :return: a Spark DataFrame or a pandas DataFrame
    """
    # TODO: add feed dictionary
    if isinstance(dframe, pd.DataFrame):
        return _map_pd(fetches, dframe, feed_dict=feed_dict, block=False, trim=None, initial_variables=initial_variables)
    return _map(fetches, dframe, feed_dict=feed_dict, block=True, trim=trim, initial_variables=initial_variables)

def reduce_blocks(fetches, dframe, initial_variables=_initial_variables_default):
    """ Applies the fetches on blocks of rows, so that only one row of data remains in the end. The order in which
    the operations are performed on the rows is unspecified.

    The `fetches` argument may be a list of graph elements or a single
    graph element. A graph element can be of the following type:

    * If the *i*th element of `fetches` is a
      `Tensor`, the *i*th return value will be a numpy ndarray containing the value of that tensor.

    There is no support for sparse tensor objects yet.

    This transform not lazy and is performed when called.

    In order to perform the reduce operation, the fetches must follow some naming conventions: for each fetch called
    for example 'z', there must be one placeholder 'z_input'. The dtype of 'z' and 'z_input' must be the same, and
    the shape of 'z_input' must be one degree higher than 'z'. For example, if 'z' is scalar, then 'z_input' must be
    a vector with unknown dimension.

    Args:
      fetches: A single graph element, or a list of graph elements
        (described above).
      dframe: A DataFrame object. The columns of the tensor frame will be fed into the fetches at execution.
    :param initial_variables: a boolean option default True, inital variables if it is used.

    Returns: a list of numpy arrays, one for each of the fetches, or a single numpy array if there is but one fetch.

    :param fetches: see description above
    :param dframe: a Spark DataFrame
    :param initial_variables: a boolean option default True, inital variables if it is used.
    :return: a list of numpy arrays
    """
    fetches = _check_fetches(fetches)
    graph = _get_graph(fetches, initial_variables)
    builder = _java_api().reduce_blocks(dframe._jdf)
    _add_graph(graph, builder)
    _add_shapes(graph, builder, fetches)
    df = builder.buildRow()
    return _unpack_row(df, fetches)

def print_schema(dframe):
    """
    Prints the schema of the dataframe, including all the metadata that describes tensor information.

    TODO: explain the data

    :param dframe: a Spark DataFrame
    :return: nothing
    """
    print(_java_api().explain(dframe._jdf))

def analyze(dframe):
    """ Analyzes a Spark DataFrame for the tensor content, and returns a new dataframe with extra metadata that
     describes the numerical shape of the content.

     This method is useful when a dataframe contains non-scalar tensors, for which the shape must be checked beforehand.

     Note: nullable fields are not accepted.

     The function [print_schema] lets users introspect the information added to the DataFrame.

    :param dframe: a Spark DataFrame
    :return: a Spark DataFrame with metadata information embedded.
    """
    return DataFrame(_java_api().analyze(dframe._jdf), _sql)

def aggregate(fetches, grouped_data, initial_variables=_initial_variables_default):
    """
    Performs an algebraic aggregation on the grouped data.


    :param fetches: a single graph element, or a sequence of graph elements
    :param grouped_data: a Spark groupedData object
    :param initial_variables: a boolean option default True, inital variables if it is used.
    :return: a dataframe, with all the columns corresponding to the fetches being appended to the
      key columns.
    """
    fetches = _check_fetches(fetches)
    graph = _get_graph(fetches, initial_variables)
    jdfin = _get_jgroup(grouped_data)
    builder = _java_api().aggregate_blocks(jdfin)
    _add_graph(graph, builder)
    _add_shapes(graph, builder, fetches)
    jdf = builder.buildDF()
    return DataFrame(jdf, _sql)

def block(df, col_name, tf_name = None):
    """
    Automatically infers a placeholder from a dataframe. The shape returned is that of blocks of
    data in the dataframe.

    The placeholder uses implicitly the current computation graph of tensorflow.

    :param df: a Spark dataframe
    :param col_name: the name of a column in a dataframe
    :param tf_name: if specified, the name assigned to the placeholder. If not specified,
    it will be the name of the column.
    :return: a TensorFlow placeholder.
    """
    return _auto_placeholder(df, col_name, tf_name, block = True)

def row(df, col_name, tf_name = None):
    """
    Automatically infers a placeholder from a dataframe. The shape returned is that of one row of
    data in the dataframe.

    The placeholder uses implicitly the current computation graph of tensorflow.

    :param df: a Spark dataframe
    :param col_name: the name of a column in a dataframe
    :param tf_name: if specified, the name assigned to the placeholder. If not specified,
    it will be the name of the column.
    :return: a TensorFlow placeholder.
    """
    return _auto_placeholder(df, col_name, tf_name, block = False)

def _auto_placeholder(df, col_name, tf_name, block):
    info = _java_api().extra_schema_info(df._jdf)
    col_shape = [x.shape() for x in info if x.fieldName() == col_name]
    if len(col_shape) == 0:
        raise Exception("Could not find column with name {col_name}")
    col_shape = col_shape[0]
    col_struct = [x for x in df.schema.fields if x.name == col_name]
    if len(col_struct) == 0:
        raise
    col_struct = col_struct[0]
    # The dtypes known to TensorFrames.
    tfdtype = _get_dtype(col_struct.dataType)
    if tf_name is None:
        tf_name = col_name
    # Use the python convention (None)
    shape = [x if x >= 0 else None for x in col_shape]
    if not block:
        shape = shape[1:]
    else:
        # The lead is always set to None, because otherwise it may choke on empty partitions.
        # (This happens when the dataset is too small)
        # TODO(tjh) add a test for this case.
        shape[0] = None
    return tf.placeholder(tfdtype, shape=shape, name=tf_name)

_dtypes = {DoubleType() : tf.double,
          IntegerType() : tf.int32,
          LongType() : tf.int64,
          FloatType() : tf.float32}

def _get_jgroup(grouped_data):
    """Get the JVM object that backs this grouped data, taking into account the different
    spark versions."""
    d = dir(grouped_data)
    if '_jdf' in d:
        return grouped_data._jdf
    if '_jgd' in d:
        return grouped_data._jgd
    raise ValueError('Could not find a dataframe for {}. All methods: {}'.format(grouped_data, d))

def _get_dtype(dtype):
    if isinstance(dtype, ArrayType):
        return _get_dtype(dtype.elementType)
    if dtype not in _dtypes:
        raise Exception("Unknown type %s " % str(dtype))
    return _dtypes[dtype]

def _validate_fetch(graph, fetch):
    try:
        fetch_t = graph.as_graph_element(fetch, allow_tensor=True,
                                         allow_operation=True)
        # For now, do not make a difference between a subfetch and a target
        return fetch_t.name
    except TypeError as e:
        raise TypeError('Fetch argument %r has invalid type %r, '
                        'must be a string or Tensor. (%s)'
                        % (fetch, type(fetch), str(e)))
    except ValueError as e:
        raise ValueError('Fetch argument %r cannot be interpreted as a '
                         'Tensor. (%s)' % (fetch, str(e)))
    except KeyError as e:
        raise ValueError('Fetch argument %r cannot be interpreted as a '
                         'Tensor. (%s)' % (fetch, str(e)))
