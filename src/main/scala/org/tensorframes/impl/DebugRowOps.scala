package org.tensorframes.impl

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.util.{Failure, Success, Try}

import org.apache.commons.lang3.SerializationUtils
import org.tensorflow.framework.GraphDef
import org.tensorflow.{Session, Tensor}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, RelationalGroupedDataset, Row}

import org.tensorframes._
import org.tensorframes.test.DslOperations


/**
 * The different schemas required for the block reduction.
 *
 * Here is the order (block reduction in a map phase followed by pair-wise aggregation in a
 * reduce phase):
 *  mapInput -> output -> reduceInput -> output
 *
 * Call 'x' a variable required by the transform, and 'y' an extra column
 *
 * @param mapInput the schema of the column block. Contains 'x_input' and 'y'
 * @param mapTFCols the indexes of the columns required by the transform
 * @param output contains 'x' only
 * @param reduceInput contains 'x_input' only ('y' column has been dropped in the map phase).
 */
case class ReduceBlockSchema(
    mapInput: StructType,
    mapTFCols: List[Int],
    output: StructType,
    reduceInput: StructType) extends Serializable


/**
  * All the schema transformations that are done by the basic TF operations.
  *
  * These methods describe the schema transforms performed on a DataFrame. They include
  * all the validation steps that should be performed before passing data to TensorFlow.
  *
  * After calling these methods, the implementation can assume the schemas are valid and complete enough.
  */
// Implementation is separated for python accessors
// TODO: these methods are pretty complicated, add more documentation!
private[impl] trait SchemaTransforms extends Logging {
  def get[A](x: Option[A], msg: String) = x.getOrElse {
    throw new Exception(msg)
  }

  def check(b: Boolean, msg: String): Unit = if (! b) {
    throw new Exception(msg)
  }


  /**
   * Validates and computes the transformation schema under reducing.
   *
   * A graph may access a subset of the rows. All the schemas are for blocks of data.
   *
   * For each output x of the graph, there must be:
   *  - a placeholder called x_input with one extra dimension (left unknown)
   *  - a corresponding column labeled 'x' with the same row structure as the output
   *
   * @param schema the schema of the dataframe
   * @param graph the graph
   * @param shapeHints the shape hints obtained for this graph
   * @return a triplet containing the input block schema, the output block schema, and the
   *         requested inputs, which may be a subset of the input.
   */
  // TODO(tjh) all the inputs and outputs are created by hashmaps, which makes their order not
  // deterministic. Change that.
  def reduceBlocksSchema(
      schema: StructType,
      graph: GraphDef,
      shapeHints: ShapeDescription): ReduceBlockSchema = {
    val summary = TensorFlowOps.analyzeGraphTF(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val fieldsByName = schema.fields.map(f => f.name -> f).toMap
    val fieldNameList = fieldsByName.keySet.toSeq.sorted.mkString(", ")
    val outputNameList = summary.filter(_._2.isOutput).keySet.toSeq.sorted.mkString(", ")
    val suffix = "_input"

    // Initial check: all the fields are here:
    val outputs = summary.filter(_._2.isOutput)

    // Check that the outputs of the graph are a subset of the columns of the dataframe.
    val missingColInputs = (outputs.keySet -- fieldsByName.keySet).toSeq.sorted
    check(missingColInputs.isEmpty, {
      val missing = missingColInputs.mkString(", ")
      s"Based on the TF graph, some inputs are missing: $missing. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList" })

    // Initial check: the inputs are all there, and they are the only ones.
    val inputs = summary.filter(_._2.isInput)
    val expectedInputs = outputs.keySet.map(_ + suffix)
    val extraInputs = (inputs.keySet -- expectedInputs).toSeq.sorted
    logDebug(s"reduceRows: expectedInputs=$expectedInputs")
    check(extraInputs.isEmpty,
      s"Extra graph inputs have been found: ${extraInputs.mkString(", ")}. " +
        s"Dataframe columns: $fieldNameList")

    val missingInputs = (expectedInputs -- inputs.keySet).toSeq.sorted
    check(missingInputs.isEmpty,
      s"Some inputs are missing in the graph: ${missingInputs.mkString(", ")}. " +
        s"Dataframe columns: $fieldNameList")

    // Check that for each output, the field is present with the right schema.
    // WARNING: keeping the order of the fields is important -> do not iterate over the outputs.
    val fields = schema.filter(f => outputs.contains(f.name)).map { f =>
      val ci = ColumnInformation(f)
      val stf = get(ci.stf,
        s"Data column '${f.name}' has not been analyzed yet, cannot run TF on this dataframe")
      // Check that the output is compatible (its presence has already been checked.
      val out = summary(f.name)
      check(out.isOutput, s"Graph node '${out.name}' should be an output")

      check(stf.dataType == out.scalarType, s"Output '${f.name}' has type ${out.scalarType}" +
        s" but the column type " +
        s"is ${stf.dataType}")

      // Take the tail, we only compare cells
      val cellShape = stf.shape.tail
      check(out.shape.checkMorePreciseThan(cellShape),
        s"Output '${f.name}' has shape ${out.shape}, not compatible with the shape " +
          s"of field elements $cellShape")
      // The input block may be too precise with respect to the lead dimension (number of rows),
      // which is usually incorrect when pairwise reductions are performed.
      // Always assume they are unknown for now.
      val shape = cellShape.prepend(Shape.Unknown)
      val inputStf = stf.copy(shape = shape)

      val inputName = f.name + suffix
      val in = get(summary.get(inputName),
        s"The graph needs to have a placeholder input called $inputName.")
      assert(in.isPlaceholder, s"Node $inputName should be a placeholder")
      assert(in.isInput, s"Node $inputName should be an input")
      check(inputStf.shape.checkMorePreciseThan(in.shape),
        s"The data column '${f.name}' has shape ${inputStf.shape}, not compatible with shape" +
          s" ${in.shape} requested by the TF graph")
      check(inputStf.dataType == in.scalarType,
        s"The type of node '${in.name}' (${inputStf.dataType}) is not compatible with the data" +
          s" type of the column (${in.scalarType})")
      val m = ColumnInformation(f, inputStf).merged
      logDebug(s">>> $m -> ${ColumnInformation(m).stf}")
      m
    }
    val outputSchema = StructType(fields.toArray)
    // The input schema is simply the block schema, with a different name for the variables.
    // We still pass all the variables because the filtering is done on the indices selected.
    val inputSchema = StructType(schema.map { f =>
      if (outputs.contains(f.name)) {
        widenLeadDim(f.copy(name = f.name + "_input"))
      } else { f }
    })
    val inputReduceSchema = StructType(schema
      .filter(f => outputs.contains(f.name))
      .map(f => widenLeadDim(f.copy(name=f.name + "_input"))))
    val requestedIndexes = schema.zipWithIndex
        .filter { case (f, idx) => outputs.contains(f.name)}
        .map(_._2) .toList
    ReduceBlockSchema(inputSchema, requestedIndexes, outputSchema, inputReduceSchema)
  }

  def reduceRowsSchema(
      schema: StructType,
      graph: GraphDef,
      shapeHints: ShapeDescription): StructType = {
    val summary = TensorFlowOps.analyzeGraphTF(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val fieldsByName = schema.fields.map(f => f.name -> f).toMap
    val fieldNameList = fieldsByName.keySet.toSeq.sorted.mkString(", ")
    val outputNameList = summary.filter(_._2.isOutput).keySet.toSeq.sorted.mkString(", ")
    val suffixes = Seq("_1", "_2")

    // Initial check: all the fields are here:
    val outputs = summary.filter(_._2.isOutput)
    // Check that there are precisely as many outputs as columns:
    if ((outputs.keySet -- fieldsByName.keySet).nonEmpty) {
      val extra = (outputs.keySet -- fieldsByName.keySet).toSeq.sorted.mkString(", ")
      val s = s"Some extra outputs were found in the reducer: $extra. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList"
      throw new Exception(s)
    }
    if ((fieldsByName.keySet -- outputs.keySet).nonEmpty) {
      val extra = (fieldsByName.keySet -- outputs.keySet).toSeq.sorted.mkString(", ")
      val s = s"Some outputs are missing in the reducer: $extra. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList"
      throw new Exception(s)
    }

    // Initial check: the inputs are all there:
    val inputs = summary.filter(_._2.isInput)
    val expectedInputs = suffixes.flatMap(suff => fieldsByName.keys.map(_ + suff)).toSet
    logDebug(s"reduceRows: expectedInputs=$expectedInputs")
    if ((inputs.keySet -- expectedInputs).nonEmpty) {
      val extra = (inputs.keySet -- expectedInputs).toSeq.sorted.mkString(", ")
      throw new Exception(
        s"Extra graph inputs have been found: $extra. Dataframe columns: $fieldNameList")
    }
    if ((expectedInputs -- inputs.keySet).nonEmpty) {
      val extra = (expectedInputs -- inputs.keySet).toSeq.sorted.mkString(", ")
      throw new Exception(
        s"Some inputs are missing in th graph: $extra. Dataframe columns: $fieldNameList")
    }

    // Check that all the fields are here
    for {
      f <- fieldsByName.values
      suffix <- suffixes
    } {
      val stf = ColumnInformation(f).stf.getOrElse { throw new Exception(
        s"Data column '${f.name}' has not been analyzed yet, cannot run TF on this dataframe")
      }
      // Check that the output is compatible (its presence has already been checked.
      val out = summary(f.name)
      if (!out.isOutput) {
        throw new Exception(
          s"Graph node '${out.name}' should be an output")
      }
      if (stf.dataType != out.scalarType) {
        val s = s"Output '${f.name}' has type ${out.scalarType} but the column type " +
          s"is ${stf.dataType}"
        throw new Exception(s)
      }
      // Take the tail, we only compare cells
      val cellShape = stf.shape.tail
      if (! out.shape.checkMorePreciseThan(cellShape)) {
        throw new Exception(
          s"Output '${f.name}' has shape ${out.shape}, not compatible with the shapes" +
            s"of field elements ${cellShape}")
      }

      // Check that the 2 inputs are compatible:
      for (suffix <- Seq("_1", "_2")) {
        val inputName = f.name + suffix
        val in = summary.getOrElse(inputName, throw new Exception(
          s"The graph needs to have a placeholder input called $inputName."))
        assert(in.isPlaceholder, s"Node $inputName should be a placeholder")
        assert(in.isInput, s"Node $inputName should be an input")
        if (! cellShape.checkMorePreciseThan(in.shape)) {
          throw new Exception(
            s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
              s" ${in.shape} requested by the TF graph")
        }
        if (stf.dataType != in.scalarType) {
          throw new Exception(
            s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data" +
              s" type of the column (${in.scalarType})")
        }
      }
    }
    // Same schema as output
    schema
  }

  // Sets the lead column to Unknown
  private def widenLeadDim(f: StructField): StructField = {
    ColumnInformation(f).stf match {
      case Some(ci) if ci.shape.numDims >= 1 =>
        val s = ci.shape.tail.prepend(Shape.Unknown)
        ColumnInformation(f, ci.copy(shape = s)).merged
      case _ => f // Nothing to do
    }
  }
}

object SchemaTransforms extends SchemaTransforms

/**
 * A simple and slow implementation of the basic operations that maximizes correctness of
 * implementation and works with older versions of Spark (based on RDDs).
 */
class DebugRowOps
  extends OperationsInterface
    with ExperimentalOperations with DslOperations with PythonInterface with Logging {


  import SchemaTransforms._

  override def toString: String = "DebugRowOps"

  override def mapBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    mapBlocks(dataframe, graph, shapeHints, appendInput = true)
  }


  override def mapBlocksTrimmed(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    mapBlocks(dataframe, graph, shapeHints, appendInput = false)
  }

  private def mapBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription,
      appendInput: Boolean): DataFrame = {
    val sc = dataframe.sqlContext.sparkContext
    val summary = TensorFlowOps.analyzeGraphTF(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val inputs = summary.filter(_._2.isInput)
    val outputs = summary.filter(_._2.isOutput)
    val fieldsByName = dataframe.schema.fields.map(f => f.name -> f).toMap
    val cols = dataframe.schema.fieldNames.mkString(", ")
    // The input schema, after validation with TF that it contains all the spark TF info.
    val inputSchema: StructType = {

      inputs.values.foreach { in =>
        val f = get(fieldsByName.get(in.name),
          s"Graph input ${in.name} found, but no column to match it. Dataframe columns: $cols")

        val stf = ColumnInformation(f).stf.getOrElse {
          throw new Exception(
            s"Data column ${f.name} has not been analyzed yet, cannot run TF on this dataframe")
        }
        // We do not support autocasting for now.
        if (stf.dataType != in.scalarType) {
          throw new Exception(
            s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data type " +
              s"of the column (${in.scalarType})")
        }
        if (! stf.shape.checkMorePreciseThan(in.shape)) {
          throw new Exception(
            s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
              s" ${in.shape} requested by the TF graph")
        }
        // The input has to be either a constant or a placeholder
        if (! in.isPlaceholder) {
          throw new Exception(
            s"Invalid type for input node ${in.name}. It has to be a placeholder")
        }
      }
      dataframe.schema
    }

    // The output schema from the data generated by TF.
    val outputTFSchema: StructType = {
      // The order of the output columns is decided for now by their names.
      val fields = outputs.values.toSeq.sortBy(_.name).map { out =>
        if (fieldsByName.contains(out.name)) {
          throw new Exception(s"TF graph has an output node called '${out.name}'," +
            s" but this column already exists. Input columns: ${cols}")
        }
        logInfo(s"mapBlocks: out = $out")
        ColumnInformation.structField(out.name, out.scalarType, out.shape)
      }
      StructType(fields.toArray)
    }
    // The column indices requested by TF
    val requestedTFInput: Array[Int] = {
      val colIdxs = dataframe.schema.fieldNames.zipWithIndex.toMap
      inputs.keys.toArray.map { name => colIdxs(name) }
    }
    // Full output schema, including data being passed through and validated for duplicates.
    // The first columns are the TF columns, followed by all the other columns.
    val outputSchema: StructType = if (appendInput) {
      StructType(outputTFSchema ++ dataframe.schema.fields)
    } else {
      StructType(outputTFSchema)
    }

    logTrace(s"mapBlocks: TF input schema = $inputSchema, complete output schema = $outputSchema")

    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      // If the partition is empty, return immediately, it seems to happen often for small datasets
      // TODO(tjh) add test for that
      if (it.hasNext) {
        DebugRowOpsImpl.performMap(
          it.toArray,
          inputSchema,
          requestedTFInput,
          gProto.value,
          outputTFSchema,
          appendInput)
      } else {
        mutable.Iterable.empty.iterator
      }
    }
    dataframe.sqlContext.createDataFrame(transformRdd, outputSchema)
  }


  override def mapRows(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    val sc = dataframe.sqlContext.sparkContext
    val summary = TensorFlowOps.analyzeGraphTF(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val inputs = summary.filter(_._2.isInput)
    val outputs = summary.filter(_._2.isOutput)
    val fieldsByName = dataframe.schema.fields.map(f => f.name -> f).toMap
    val cols = dataframe.schema.fieldNames.mkString(", ")

    // Initial check of the input.
    inputs.values.foreach { in =>
      val fname = get(shapeHints.inputs.get(in.name),
        s"The graph placeholder ${in.name} was not given a corresponding dataframe field name as input:" +
          s"hints: ${shapeHints.inputs}")

      val f = get(fieldsByName.get(fname),
        s"Graph input ${in.name} found, but no column to match it. Dataframe columns: $cols")

      val stf = get(ColumnInformation(f).stf,
        s"Data column ${f.name} has not been analyzed yet, cannot run TF on this dataframe")

      check(stf.dataType == in.scalarType,
        s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data type " +
          s"of the column (${in.scalarType})")

      val cellShape = stf.shape.tail
      // No check for unknowns: we allow unknowns in the first dimension of the cell shape.
      check(cellShape.checkMorePreciseThan(in.shape),
        s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
          s" ${in.shape} requested by the TF graph")

      check(in.isPlaceholder,
        s"Invalid type for input node ${in.name}. It has to be a placeholder")
    }

    // The output schema of the block from the data generated by TF.
    val outputTFSchema: StructType = {
      // The order of the output columns is decided for now by their names.
      val fields = outputs.values.toSeq.sortBy(_.name).map { out =>
        check(! fieldsByName.contains(out.name),
          s"TF graph has an output node called '${out.name}'," +
            s" but this column already exists. Input columns: ${cols}")
        // The shapes we get in each output node are the shape of the cells of each column, not the
        // shape of the column. Add Unknown since we do not know the exact length of the block.
        val blockShape = out.shape.prepend(Shape.Unknown)
        ColumnInformation.structField(out.name, out.scalarType, blockShape)
      }
      StructType(fields.toArray)
    }

    // The column indices requested by TF, and the name of the placeholder that gets fed.
    val requestedTFInput: Array[(NodePath, Int)] = {
      val colIdxs = dataframe.schema.fieldNames.zipWithIndex.toMap
      inputs.keys.map { nodePath =>
        val fieldName = shapeHints.inputs(nodePath)
        nodePath -> colIdxs(fieldName)
      }   .toArray
    }
    // Full output schema, including data being passed through and validated for duplicates.
    // The first columns are the TF columns, followed by all the other columns.
    val outputSchema: StructType = {
      StructType(outputTFSchema ++ dataframe.schema.fields)
    }

    val schema = dataframe.schema // Classic rookie mistake...
    logTrace(s"mapRows: input schema = $schema, requested cols: ${requestedTFInput.toSeq}" +
      s" complete output schema = $outputSchema")
    // TODO: this is leaking the file.
    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      DebugRowOpsImpl.performMapRows(
        it.toArray,
        schema,
        requestedTFInput,
        gProto.value,
        outputTFSchema).toIterator
    }
    dataframe.sqlContext.createDataFrame(transformRdd, outputSchema)
  }

  override def reduceRows(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = {
    val sc = dataframe.sqlContext.sparkContext
    // Most of this function is various sanity checks on the graph and on the dataframe.
    reduceRowsSchema(dataframe.schema, graph, shapeHints)
    val schema = dataframe.schema
    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      // TODO(tjh) add test for empty iterator
      val arr = it.toArray
      if (arr.length <= 1) {
        arr.iterator
      } else {
        val row = DebugRowOpsImpl.performReducePairwise(
          arr, schema, gProto.value
        )
        Array(row).iterator
      }
    }
    transformRdd.reduce(DebugRowOpsImpl.reducePair(schema, gProto))
  }

  override def reduceBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = {
    val sc = dataframe.sqlContext.sparkContext
    val schema = dataframe.schema
    val allSchema = reduceBlocksSchema(schema, graph, shapeHints)
    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    // It first reduces each block, and then performs pair-wise reduction.
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      // TODO(tjh) add test for empty iterator
      val arr = it.toArray
      if (arr.length <= 1) {
        arr.iterator
      } else {
        val row = DebugRowOpsImpl.performReduceBlock(
          arr, allSchema.mapInput, allSchema.mapTFCols, allSchema.output, gProto.value)
        Array(row).iterator
      }
    }
    // TODO: it would be nice to respect the output order here?
    transformRdd.reduce(DebugRowOpsImpl.reducePairBlock(
      allSchema.reduceInput, allSchema.output, gProto))
  }

  override def explain(df: DataFrame): String = {
    val d = explainDetailed(df)

    val builder = new StringBuilder
    builder.append("root\n")
    val prefix = " |"
    d.cols.foreach { col =>
      val f = col.field
      builder.append(s"$prefix-- ${f.name}: ${f.dataType.typeName} (nullable = ${f.nullable})")
      val stf = col.stf.map { s =>
        val dt = SupportedOperations.opsFor(s.dataType).sqlType
        s" ${dt.typeName}${s.shape}"
      }   .getOrElse(" <no tensor info>")
      builder.append(stf)
      builder.append("\n")
    }
    builder.toString()
  }

  override def aggregate(
      data: RelationalGroupedDataset,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    // The constraints on the graph are the same as blocked data.
    val dataframe = DebugRowOpsImpl.backingDF(data) match {
      case Success(d) => d
      case Failure(e) =>
        throw e
    }
    logDebug(s"aggregate: found dataframe: $dataframe")

    val sc = dataframe.sqlContext.sparkContext

    val allSchemas = SchemaTransforms.reduceBlocksSchema(
      dataframe.schema, graph, shapeHints)
    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))

    // TODO(tjh) fix the size of the schema based on the overall size of the row, which we know
    // based on the shapes (there is no unknown shape when reducing).
    // The input schema is a bit different than for reduce: we already filter the columns we are
    // interested in.
    val inputSchema = StructType(allSchemas.output.map { of =>
      allSchemas.mapInput(of.name + "_input")
    })

    val tfudaf = new TensorFlowUDAF(allSchemas.output, inputSchema, gProto, 10)
    val cols = allSchemas.output.fields.map(f => col(f.name))
    val sname = "tf_output"
    val df2 = data.agg(tfudaf(cols: _*).as(sname))
    // Unpack the structure as the main columns:
    // For some reason, the metadata does not get through, so it is added here as a post-processing
    // step.
    // TODO(tjh) add a test for shape propagation
    val unpackCols = allSchemas.output.map { field =>
      val n = field.name
      val meta = ColumnInformation(field).merged.metadata
      logDebug(s"aggregate: output: $n -> $meta")
      col(s"$sname.$n").as(n, meta)
    }

    // The columns that were used for the key
    val othercols = df2.schema.fieldNames.filter(_ != sname).map(n => col(n))
    val allcols = othercols ++ unpackCols
    df2.select(allcols: _*)
  }
}

/**
 * Simple implementation of a reduction with TF graphs.
 *
 * This version keeps a running buffer of elements and compacts them once the buffer becomes too
 * big or once the evaluation is called.
 */
class TensorFlowUDAF(
    val rowSchema: StructType,
    val tfInputSchema: StructType,
    val gProto: Broadcast[SerializedGraph],
    val bufferSize: Int) extends UserDefinedAggregateFunction with Logging {

  private val COUNT = 0
  private val ROWS = 1
  private type RowArray = mutable.WrappedArray[Row]

  override def inputSchema: StructType = rowSchema

  // We keep a buffer of rows, a counter of how many rows we have seen so far, and the current
  // aggregates before operating on the given rows.
  override val bufferSchema: StructType = {
    StructType(List(
      StructField("counter", IntegerType, nullable = false),
      StructField("rows", ArrayType(rowSchema, containsNull = true), nullable = false)
    ))
  }

  override def deterministic: Boolean = true

  override def dataType: DataType = rowSchema

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(COUNT) = 0
    buffer(ROWS) = empty
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val currentCount = count(buffer)
    buffer(COUNT) = currentCount + 1
    val arr = array(buffer)
    logTrace(s"update: arr = $arr")
    arr(currentCount) = input.copy()
    logTrace(s"update: arr2 = $arr")
    buffer(ROWS) = arr
    if (currentCount >= bufferSize) {
      compact(buffer)
    }
  }

  private def compact(buffer: MutableAggregationBuffer): Unit = {
    buffer(COUNT) = 1
    val arr0 = array(buffer)
    val arr1 = empty
    arr1(0) = compact2(arr0)
    buffer(ROWS) = arr1
  }

  private def compact2(arr: mutable.WrappedArray[Row]): Row = {
    // This is simply performing a block reduce on the rows. The schema has already
    DebugRowOpsImpl.performReduceBlock(
      arr.toArray.filterNot(_ == null), tfInputSchema, rowSchema.indices, rowSchema, gProto.value)
  }

  override def merge(buffer: MutableAggregationBuffer, other: Row): Unit = {
    logTrace(s"merge: buffer=$buffer, other=$other")
    val initialCount = count(buffer)
    val otherCount = count(other)
    val (currentCount, totalCount) = if (initialCount + otherCount >= bufferSize) {
      compact(buffer)
      (1, 1 + otherCount)
    } else {
      (initialCount, initialCount + otherCount)
    }
    buffer(COUNT) = totalCount
    val arr = array(buffer)
    val arr2 = array(other)
    for (i <- 0 until otherCount) {
      arr(currentCount + i) = arr2(i).copy()
    }
    buffer(ROWS) = arr
    logTrace(s"merge (after): buffer=$buffer")
  }

  override def evaluate(buffer: Row): Row = {
    logTrace(s"evaluate: $buffer")
    val c = count(buffer)
    require(c >= 1, buffer)
    // No need for compaction
    if (c == 1) {
      array(buffer).head
    } else {
      compact2(array(buffer))
    }
  }

  private def count(b: Row): Int = b.getInt(COUNT)
  private def array(buffer: Row): RowArray = {
    buffer.getAs[RowArray](ROWS)
  }
  private def empty: RowArray = Array.fill[Row](bufferSize + 1)(null)
}

object DebugRowOpsImpl extends Logging {

  /**
   * Accesses the backing dataframe by reflection.
   *
   * This is very brittle, there should be some other ways of doing it.
   *
   * @param groupedData the grouped data
   * @return the dataframe, if it succeeded.
   */
  def backingDF(groupedData: RelationalGroupedDataset): Try[DataFrame] = {
      Try {
        groupedData.getClass.getDeclaredMethods.foreach { m =>
          logDebug(s"method: ${m.getName}")
        }
        val method = groupedData.getClass.getDeclaredMethod("df")
        method.setAccessible(true)
        method.invoke(groupedData).asInstanceOf[DataFrame]
      }   .orElse {
        Try {
          // Find the name of the field, which is tricky...
          val fname = groupedData.getClass
            .getDeclaredFields.find(_.getName.endsWith("df")).getOrElse {
            groupedData.getClass.getDeclaredFields.foreach { m =>
              logDebug(s"field: ${m.getName}")
            }
            throw new Exception("Could not find field")
          }
          val method = groupedData.getClass.getDeclaredField(fname.getName)
          method.setAccessible(true)
          method.get(groupedData).asInstanceOf[DataFrame]
        }
      }
  }

  private[impl] def reducePair(
      schema: StructType,
      gbc: Broadcast[SerializedGraph]): (Row, Row) => Row = {
    def f(row1: Row, row2: Row): Row = {
      performReducePairwise(Array(row1, row2), schema, gbc.value)
    }
    f
  }

  def reducePairBlock(
      inputSchema: StructType,
      outputSchema: StructType,
      gbc: Broadcast[SerializedGraph]): (Row, Row) => Row = {
    def f(row1: Row, row2: Row): Row = {
      performReduceBlock(Array(row1, row2), inputSchema, inputSchema.indices.toArray, outputSchema,
        gbc.value)
    }
    f
  }
  /**
   * Performs the data transform. All the data has been verified at this point.
   * 
   * The output format is: all the columns coming out of TF first (their exact content being given
   * by tfOutputSchema), and then all the input columns.
   *
   * @param input the row of input data
   * @param inputSchema the schema of the block of inputs
   * @param inputTFCols the columns of input data that will be converted to TF data and fed into the
   *                    TF session
   * @param graphDef the graph definition that contains the description of the TF graph
   * @param tfOutputSchema the (expected) output schema of the block of outputs
   * @return the array of rows that combine outputs from the TF graph with all the other data
   *         columns.
   */
  def performMap(
      input: Array[Row],
      inputSchema: StructType,
      inputTFCols: Array[Int],
      graphDef: SerializedGraph,
      tfOutputSchema: StructType,
      appendInput: Boolean): Iterator[Row] = {
    logDebug(s"performMap: inputSchema=$inputSchema, tfschema=$tfOutputSchema," +
      s" ${input.length} rows, input cols: ${inputTFCols.toSeq}")
    // Some partitions may be empty
    // TODO(tjh) add test for that
    if (input.length == 0) {
      return Iterator.empty
    }
    // Force the content to be sent to disk. This happens in the workers, so it should be a safe operation.
    graphDef.evictContent()

    TensorFlowOps.withSession(graphDef) { session =>
      val inputTensors = TFDataOps.convert(input, inputSchema, inputTFCols)
      logDebug(s"performMap:inputTensors=$inputTensors")
      val requested = tfOutputSchema.map(_.name)
      var runner = session.runner()
      for (req <- requested) {
        runner = runner.fetch(req)
      }
      for ((inputName, inputTensor) <- inputTensors) {
        runner = runner.feed(inputName, inputTensor)
      }
      val outs = runner.run().asScala
      logDebug(s"performMap:outs=$outs")
      // Close the inputs
      inputTensors.map(_._2).foreach(_.close())
      val res = TFDataOps.convertBack(outs, tfOutputSchema, input, inputSchema, appendInput)
      // Close the outputs
      outs.foreach(_.close())
      res
    }
  }

  // For testing only:
  def performMap(
      input: Array[Row],
      inputSchema: StructType,
      inputTFCols: Array[Int],
      graphDef: GraphDef,
      outputSchema: StructType): Array[Row] = {
    // Do a defensive copy of the content of the iterator, as the object may be reused.
    performMap(input, inputSchema, inputTFCols,
      TensorFlowOps.graphSerial(graphDef), outputSchema, appendInput = true).map { row =>
      SerializationUtils.clone(row)
    } .toSeq.toArray
  }

  def performMapRows(
      input: Array[Row],
      inputSchema: StructType,
      inputTFCols: Array[(NodePath, Int)],
      graphDef: SerializedGraph,
      tfOutputSchema: StructType): Array[Row] = {
    // Some partitions may be empty
    if (input.length == 0) {
      return Array.empty
    }
    // Force the content to be sent to disk. This happens in the workers, so it should be a safe operation.
    graphDef.evictContent()

    TensorFlowOps.withSession(graphDef) { session =>
      input.map { row =>
        val inputTensors = TFDataOps.convert(row, inputSchema, inputTFCols)
        logDebug(s"performMap:inputTensors=$inputTensors")
        val requested = tfOutputSchema.map(_.name)
        var runner = session.runner()
        for (req <- requested) {
          runner = runner.fetch(req)
        }
        for ((inputName, inputTensor) <- inputTensors) {
          runner = runner.feed(inputName, inputTensor)
        }
        val outs = runner.run().asScala
        logDebug(s"performMap:outs=$outs")
        // Close the inputs
        inputTensors.map(_._2).foreach(_.close())
        val res = TFDataOps.convertBack(outs, tfOutputSchema, Array(row), inputSchema, appendInput = true)
        // Close the outputs
        outs.foreach(_.close())
        assert(res.hasNext)
        val r = res.next()
        assert(!res.hasNext)
        r
      }
    }
  }

  /**
   * Performs a reduce operation on the set of columns.
   *
   * This performs a fix-point operation around the given schema:
   *  - all the columns are transformed
   *  - for each output X there is a given X_input input. The schema is assumed to be the same with
   *  an extra block dimension.
   *
   * @param input the input of rows.
   * @param schema the output schema.
   * @param graphDef the definition of the graph
   * @return the combined row after transform
   */
  def performReduceBlock(
      input: Array[Row],
      inputSchema: StructType,
      inputTFCols: Seq[Int],
      schema: StructType,
      graphDef: SerializedGraph): Row = {
    logDebug(s"performReduceBlock: schema=$schema inputSchema=$inputSchema with ${input.length} rows")
    // The input schema and the actual data representation depend on the block operation.

    // Evict the graph to disk to free up some memory.
    TensorFlowOps.withSession(graphDef) { session =>
      val inputTensors = TFDataOps.convert(input, inputSchema, inputTFCols.toArray)
      logDebug(s"performReduceBlocks: inputTensors=$inputTensors")
      val requested = schema.map(_.name)
      val outputs = performRunner(session, requested, inputTensors)
      val emptyRows = Array.fill(1)(emptyRow)
      val it = TFDataOps.convertBack(outputs, schema, emptyRows, emptySchema, appendInput = false)
      assert(it.hasNext)
      val r = it.next()
      assert(!it.hasNext, it.next())
      outputs.foreach(_.close())
      r
    }
  }

  /**
    * Performs a run, and deallocates the inputs, regardless of the success of the run.
    */
  private def performRunner(
      session: Session,
      requested: Seq[String],
      inputs: Seq[(String, Tensor)]): Seq[Tensor] = {
    var runner = session.runner()
    for (req <- requested) {
      runner = runner.fetch(req)
    }
    for ((inputName, inputTensor) <- inputs) {
      runner = runner.feed(inputName, inputTensor)
    }
    try {
      runner.run().asScala
    } finally {
      inputs.map(_._2).foreach(_.close())
    }

  }

  private val emptySchema = StructType(Seq.empty)
  private val emptyRow: Row = new GenericRowWithSchema(Array.empty[Any], emptySchema)

  /**
   * Performs a reduce operation on a set of columns by performing pair-wise reductions.
   *
   * @param input
   * @param schema
   * @param graphDef
   * @return
   */
  def performReducePairwise(
      input: Array[Row],
      schema: StructType,
      graphDef: SerializedGraph): Row = {

    // Dump the graph onto disk if necessary
    graphDef.evictContent()

    require(input.length > 0, "Cannot provide empty input")
    // If there is a single row, no need to perform operations.
    if (input.length == 1) {
      return input.head
    }
    val inputSchema = {
      val f1 = schema.fields.map(f => f.copy(name = s"${f.name}_1"))
      val f2 = schema.fields.map(f => f.copy(name = s"${f.name}_2"))
      StructType(f1 ++ f2)
    }
    // For efficiency, this tries to reuse the session.
    val tfInput: Array[Row] = input.tail
    var result: Row = input.head

    TensorFlowOps.withSession(graphDef) { session =>
      for (row <- tfInput) {
        val values = (result.toSeq ++ row.toSeq).toArray
        val r: Row = new GenericRowWithSchema(values, inputSchema)
        val thisInput = Array(r)
        val inputs = TFDataOps.convert(thisInput, inputSchema, inputSchema.fields.indices.toArray)
        val requested = schema.map(_.name)
        val outputs = performRunner(session, requested, inputs)
        val emptyRows = Array.fill(1)(emptyRow)
        val it = TFDataOps.convertBack(outputs, schema, emptyRows, emptySchema, appendInput = false)
        assert(it.hasNext)
        result = it.next()
        assert(!it.hasNext)
        outputs.foreach(_.close())
      }
    }
    result
  }
}

object DebugRowOps extends DebugRowOps
