package org.tensorframes.impl

import scala.collection.mutable
import scala.reflect.ClassTag
import org.bytedeco.javacpp.{tensorflow => jtf}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.{GenericRow, GenericRowWithSchema}
import org.apache.spark.sql.types.{NumericType, StructType}
import org.tensorframes.{ColumnInformation, Logging, NodePath, Shape}
import org.tensorframes.Shape.DimType

/**
  * Converts data between the C++ runtime of TensorFlow and the Spark runtime.
 *
 * Current (only) implementation exports each row and copies it back into a C++ buffer.
 */
object DataOps extends Logging {

  /**
   * (Slow) implementation that takes data in C++ and puts it back into SQL rows, following
   * the structure provided and merging back all the columns from the input.
    *
    * @param tv
   * @param tf_struct the structure of the block represented in TF
   * @return an iterator that lazily computes the rows back.
   */
  // Note that doing it this way is very inefficient, but columnar implementation should prevent all this
  // data copying in most cases.
  // TODO PERF: the current code allocates a new row for each of the rows returned.
  // Instead of doing that, it could allocate once the memory and reuse the same rows and objects.
  def convertBack(
      tv: jtf.TensorVector,
      tf_struct: StructType,
      input: Array[Row],
      input_struct: StructType,
      appendInput: Boolean,
      fastPath: Boolean = true): Iterator[Row] = {
    // The structures should already have been validated.
    // Output has all the TF columns first, and then the other columns
    logDebug(s"convertBack: ${input.length} input rows, tf_struct=$tf_struct")

    val tfSizesAndIters = for ((t, idx) <- tf_struct.fields.zipWithIndex.toSeq) yield {
      val info = ColumnInformation(t).stf.getOrElse {
        throw new Exception(s"Missing info in field $t")
      }
      logTrace(s"convertBack: $t $info")
      // Drop the first cell, this is a block.
      val expLength = if (appendInput) { Some(input.length) } else { None }
      val (numRows, iter) = getColumn(tv, idx, info.dataType, info.shape.tail, expLength)
      numRows -> iter
    }
    val tfSizes = tfSizesAndIters.map(_._1)
    val tfNumRows: Int = tfSizes.distinct match {
      case Seq(x) => x
      case Seq() =>
        throw new Exception(s"Output cannot be empty. tf_struct=$tf_struct")
      case _ =>
        throw new Exception(s"Multiple number of rows detected. tf_struct=$tf_struct," +
          s" tfSizes = $tfSizes")
    }
    assert((!appendInput) || tfNumRows == input.length,
      s"Incompatible sizes detected: appendInput=$appendInput, tf num rows = $tfNumRows, " +
        s"input num rows = ${input.length}")
    val tfIters = tfSizesAndIters.map(_._2.iterator).toArray
    val outputSchema = if (appendInput) {
      StructType(tf_struct.fields ++ input_struct.fields)
    } else {
      StructType(tf_struct.fields)
    }
    val res: Iterator[Row] = if (fastPath) {
      convertBackFast0(input, tfIters, tfNumRows, input_struct, outputSchema)
    } else {
      convertBackSlow0(input, tfIters, input_struct, outputSchema)
    }
    logTrace(s"outputSchema=$outputSchema")
    logTrace(s"res: $res")
    res
  }

  // TODO(tjh) remove this method, it is buggy now for appendInput = false
  private[this] def convertBackSlow0(
      input: Array[Row],
      tfIters: Array[Iterator[Any]],
      input_struct: StructType,
      outputSchema: StructType): Iterator[Row] = {
    val allIters: Array[Iterator[Any]] = {
      if (input_struct.isEmpty) {
        tfIters
      } else {
        val riters = input_struct.indices.map (idx => getColumn(input, idx).iterator)
        tfIters ++ riters
      }
    }
    val res = for (i <- input.indices) yield {
      assert(allIters.forall(_.hasNext))
      val current = allIters.map(_.next())
      new GenericRowWithSchema(current, outputSchema)
    }
    res.iterator
  }

  private[this] def convertBackFast0(
      input: Array[Row],
      tfIters: Array[Iterator[Any]],
      tfNumRows: Int,
      input_struct: StructType,
      outputSchema: StructType): Iterator[Row] = {
    val numOutCols = outputSchema.size
    val numInCols = input_struct.size
    val numTFCols = tfIters.length
    // We check if we need to append the input columns to the final output.
    val appendInput = numOutCols == numInCols + numTFCols
    assert(
      numOutCols == numInCols + numTFCols || numOutCols == numTFCols,
      (numOutCols, numInCols, numTFCols))
    val numRowsToProcess = if (appendInput) {
      assert(input.length == tfNumRows, (input.length, tfNumRows))
      input.length
    } else { tfNumRows }
    val res: Array[GenericRow] = new Array[GenericRow](numRowsToProcess)
    var rowIdx = 0
    while(rowIdx < numRowsToProcess) {
      val rowContent = new Array[Any](numOutCols)
      // Transfer the content of the TF outputs
      var tfColIdx = 0
      while (tfColIdx < tfIters.length) {
        rowContent(tfColIdx) = tfIters(tfColIdx).next()
        tfColIdx += 1
      }
      if (appendInput) {
        // Copy the existing row into the output row
        val r = input(rowIdx)
        var colIdx = 0
        while (colIdx < numInCols) {
          rowContent(numTFCols + colIdx) = r.get(colIdx)
          colIdx += 1
        }
      }
      res(rowIdx) = new GenericRow(rowContent)
      rowIdx += 1
    }
    res.iterator
  }

  /**
   * Performs size checks and resolutions, and converts the data from the row format to the C++
   * buffers.
   *
   * @param it
   * @param struct the structure of the block. It should contain all the extra meta-data required by
   *               TensorFrames.
   * @param requestedTFCols: the columns that will be fed into TF
   * @return
   */
  def convert(
      it: Array[Row],
      struct: StructType,
      requestedTFCols: Array[Int],
      fastPath: Boolean = true): jtf.StringTensorPairVector = {
    // This is a very simple and very inefficient implementation. It should be kept
    // as is for correctness checks.
    val fields = requestedTFCols.map(struct.fields(_))
    logDebug(s"Calling convert on ${it.length} rows with struct: $struct " +
      s"and indices: ${requestedTFCols.toSeq}")
    val converters = fields.map { f =>
      // Extract and check the shape
      val ci = ColumnInformation(f).stf.getOrElse {
        throw new Exception(s"Could not column information for column $f")
      }
      val leadDim = ci.shape.dims.headOption.getOrElse {
        throw new Exception(s"Column $f found to be scalar, but its dimensions should be >= 1")
      } .toInt
      if (leadDim != Shape.Unknown && leadDim != it.length) {
        throw new Exception(s"Lead dimension for column $f (found to be $leadDim)" +
          s" is not compatible with a block of size ${it.length}. " +
          s"Expected block structure: $struct, meta info = $ci")
      }
      SupportedOperations.opsFor(ci.dataType).tfConverter(ci.shape.tail, it.length)
    }
    for (c <- converters) { c.reserve() }

    if (fastPath) {
      convertFast0(it, converters, requestedTFCols)
    } else {
      convertSlow0(it, converters, requestedTFCols)
    }

    val tensors = converters.map(_.tensor())
    val names = requestedTFCols.map(struct(_).name)
    for ((name, t) <- names.zip(tensors)) {
      logTrace(s"convert: $name : ${TensorFlowOps.jtfShape(t.shape())}")
    }
    new jtf.StringTensorPairVector(names, tensors)
  }

  private[this] def convertSlow0(
      it: Array[Row],
      converters: Array[TensorConverter[_]],
      requestedTFCols: Array[Int]): Unit = {
    for (r <- it) {
      for ((c, idx) <- converters.zip(requestedTFCols)) {
        c.append(r, idx)
      }
    }
  }

  private[this] def convertFast0(
      it: Array[Row],
      converters: Array[TensorConverter[_]],
      requestedTFCols: Array[Int]): Unit = {
    // Unrolled for performance
    val numRows = it.length
    val numRequestedCols = requestedTFCols.length
    var requestedColIdx = 0
    while (requestedColIdx < numRequestedCols) {
      val converter = converters(requestedColIdx)
      val colIdx = requestedTFCols(requestedColIdx)
      var rowIdx = 0
      while(rowIdx < numRows) {
        converter.append(it(rowIdx), colIdx)
        rowIdx += 1
      }
      requestedColIdx += 1
    }
  }

  /**
    * Converts a single row at a time.
    *
    * @param r the row to convert
    * @param blockStruct the structure of the block that produced this row
    * @param requestedTFCols the requested columns
    * @return
    */
  def convert(
      r: Row,
      blockStruct: StructType,
      requestedTFCols: Array[(NodePath, Int)]): jtf.StringTensorPairVector = {
    // This is a very simple and very inefficient implementation. It should be kept
    // as is for correctness checks. The columnar implementation is meant to be more
    // efficient.
    logDebug(s"Calling convert on one with struct: $blockStruct")
    val elts = requestedTFCols.map { case (npath, idx) =>
      val f = blockStruct.fields(idx)
      // Extract and check the shape
      val ci = ColumnInformation(f).stf.getOrElse {
        throw new Exception(s"Could not column information for column $f")
      }
      assert(ci.shape.numDims >= 1,
        s"Column $f found to be a scala, but its dimensions should be >= 1")
      // Special case: if the cell shape has undefined size in its first argument, we
      // still accept it and attempt to get it from the shape. This is to support rows
      // with different vector sizes. All other dimensions must match, although this
      // could be relaxed in the future as well. It is harder to check.
      val cellShape = {
        val givenCellShape = ci.shape.tail
        if (givenCellShape.dims.headOption == Some(Shape.Unknown)) {
          r.get(idx) match {
            case s: Array[_] =>
              givenCellShape.tail.prepend(s.length.toLong)
            case s: Seq[_] =>
              givenCellShape.tail.prepend(s.length.toLong)
            case _ => givenCellShape
          }
        } else {
          givenCellShape
        }
      }
      assert(!cellShape.hasUnknown,
        s"The computed shape for the cell $idx (field $f) is $cellShape, which has unknowns")

      val conv = SupportedOperations.opsFor(ci.dataType).tfConverter(cellShape, 1)
      conv.reserve()
      conv.append(r, idx)
      npath -> conv.tensor()
    }
    val nodePaths = elts.map(_._1)
    val tensors = elts.map(_._2)
    new jtf.StringTensorPairVector(nodePaths, tensors)
  }

  // **** Conversions TF => Spark ***

  private[this] def reshapeIter[T : ClassTag](
      it: mutable.WrappedArray[T],
      s: List[DimType]): Seq[Any] = {
    // The reshaping is extremely inefficient but the output is easy to read after that.
    s match {
      case Nil => throw new Exception()
      case n :: Nil =>
        val x = it.toArray.toSeq
        assert(x.size == n.toInt, s"$n $x")
        x
      case n :: t =>
        val blockSize = t.product.toInt
        val x = it.grouped(blockSize).map(reshapeIter(_, t)).toArray.toSeq
        assert(x.size == n.toInt, s"$n:$t $it")
        x
    }
  }

  @throws[IllegalArgumentException]("If the shape contains unknown and the expected number of " +
    "rows is not provided")
  private def inferPhysicalShape(
      numScalars: Int,
      expectedCellShape: Shape,
      expectedNumRows: Option[Int]): (Int, Shape) = {
    if (expectedCellShape.dims.count(_ == Shape.Unknown) > 1) {
      throw new IllegalArgumentException(s"Shape has too many unkown values to perform inference: " +
        s"$expectedCellShape")
    }
    val h = expectedCellShape.dims.takeWhile(_ >= 0)
    val t = expectedCellShape.dims.dropWhile(_ >= 0) match {
      case Seq() => Nil
      case x: Seq[DimType] => x.tail
    }
    val p = h.product * t.product
    expectedNumRows match {
      case None if expectedCellShape.hasUnknown =>
        // Underresolved
        throw new IllegalArgumentException(s"Cannot infer the final cell shape: $expectedCellShape")
      case Some(numRows) if ! expectedCellShape.hasUnknown =>
        // Overresolved, just check everything is compatible.
        if (numScalars != p * numRows) {
          throw new IllegalArgumentException(s"Expected ${p * numRows} elements in the final " +
            s"buffer," +
            s"but got $numScalars instead. shape=$expectedCellShape, numRows=$numRows")
        }
        numRows -> expectedCellShape
      case Some(numRows) =>
        // Compute the missing shape value
        val inferred = numScalars / (p * numRows)
        val inferredShape = Shape(((h :+ inferred.toLong) ++ t).toArray)
        assert(inferredShape.dims.product * numRows == numScalars,
          s"Incompatible values: shape=$expectedCellShape numRows=$numRows numScalars=$numScalars")
        numRows -> inferredShape
      case None =>
        val numRows = (numScalars / p).toInt
        assert(expectedCellShape.dims.product * numRows == numScalars,
          s"Incompatible values: shape=$expectedCellShape numRows=$numRows numScalars=$numScalars")
        numRows -> expectedCellShape
    }
  }

  /**
   * Extracts the content of a column as objects amenable to SQL.
    *
    * @param tv
   * @param position
   * @param scalaType the scalar type of the tensor
   * @param cellShape the shape of each cell of data
   * @param expectedNumRows the expected number of rows in the output. Depending on the shape
   *                        (which may have unknowns) and the expected number of rows (which may
   *                        also be unknown), this function will try to compute both the physical
   *                        shape and the actual number of rows based on the size of the
   *                        flattened tensor.
   * @return the number of rows and an iterable over the rows
   */
  private def getColumn(
      tv: jtf.TensorVector,
      position: Int,
      scalaType: NumericType,
      cellShape: Shape,
      expectedNumRows: Option[Int],
      fastPath: Boolean = true): (Int, Iterable[Any]) = {
    val t = tv.get(position)
    logTrace(s"getColumn: shape: ${TensorFlowOps.jtfShape(t.shape())}  " +
      s"cellShape:$cellShape numRows:$expectedNumRows")
    logTrace(s"getColumn: got tensor: ${t.DebugString().toString}")
    val rawBuff = t.tensor_data().asBuffer()
    val allDataBuffer: mutable.WrappedArray[_] =
      SupportedOperations.opsFor(scalaType).convertBuffer(rawBuff)
    val numData = allDataBuffer.size
    // Infer if necessary the reshaping size.
    val (inferredNumRows, inferredShape) = inferPhysicalShape(numData, cellShape, expectedNumRows)
    logTrace(s"getColumn: databuffer = $allDataBuffer")
    logTrace(s"getColumn: infered cell shape: $inferredShape, numData: $numData," +
      s" inferredNumRows: $inferredNumRows")
    val reshapeShape = inferredShape.prepend(inferredNumRows)
    val res = if (fastPath) {
      getColumnFast0(reshapeShape, scalaType, allDataBuffer)
    } else {
      reshapeIter(allDataBuffer.asInstanceOf[mutable.WrappedArray[Any]],
        inferredShape.dims.toList)
    }
    // The old implementation
    logTrace(s"getColumn: reshaped = $res")
    inferredNumRows -> res
  }

  private[this] def getColumnFast0(
      reshapeShape: Shape,
      scalaType: NumericType,
      allDataBuffer: mutable.WrappedArray[_]): Iterable[Any] = {
    reshapeShape.dims match {
      case Seq() =>
        throw new AssertionError(s"dims should not be empty")
      case Seq(dim1) =>
        SupportedOperations.opsFor(scalaType)
          .convertBuffer1(allDataBuffer.array, dim1.toInt)
      case Seq(dim1, dim2) =>
        SupportedOperations.opsFor(scalaType)
          .convertBuffer2(allDataBuffer.array, dim1.toInt, dim2.toInt)
      case Seq(dim1, dim2, dim3) =>
        SupportedOperations.opsFor(scalaType)
          .convertBuffer3(allDataBuffer.array, dim1.toInt, dim2.toInt, dim3.toInt)
      case x: Seq[_] =>
        throw new IllegalArgumentException(s"Operations for tensors of order ${x.size}" +
          s" are not supported")
    }
  }

  private def getColumn(rows: Iterable[Row], position: Int): Iterable[Any] = {
    rows.map(_.get(position))
  }
}
