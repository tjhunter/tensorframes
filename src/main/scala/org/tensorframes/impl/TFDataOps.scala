package org.tensorframes.impl

import scala.collection.mutable
import org.{tensorflow => tf}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{NumericType, StructType}
import org.tensorframes.{ColumnInformation, Logging, NodePath, Shape}

/**
 * Converts data between the C++ runtime of TensorFlow and the Spark runtime.
 *
 * Current (only) implementation exports each row and copies it back into a C++ buffer.*
 * This implementation uses the official Java Tensorflow API (experimental).
 */
object TFDataOps extends Logging {

  /**
    * Performs size checks and resolutions, and converts the data from the row format to the C++
    * buffers.
    *
    * @param it
    * @param struct the structure of the block. It should contain all the extra meta-data required by
    *               TensorFrames.
    * @param requestedTFCols: the columns that will be fed into TF
    * @return pairs of plaholder path -> input tensor
    */
  def convert(
      it: Array[Row],
      struct: StructType,
      requestedTFCols: Array[(NodePath, Int)]): Seq[(String, tf.Tensor)] = {
    // This is a very simple and very inefficient implementation. It should be kept
    // as is for correctness checks.

    val convertersWithPaths = requestedTFCols.map { case (npath, idx) =>
      val f = struct.fields(idx)
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
      val conv = SupportedOperations.opsFor(ci.dataType).tfConverter(ci.shape.tail, it.length)
      conv.reserve()
      npath -> conv
    }
    val converters = convertersWithPaths.map(_._2)
    // The indexes requested by tensorflow
    val requestedTFColIdxs = requestedTFCols.map(_._2)
    DataOps.convertFast0(it, converters, requestedTFColIdxs)

    convertersWithPaths.map { case (npath, conv) => npath -> conv.tensor2() }
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
      requestedTFCols: Array[(NodePath, Int)]): Seq[(String, tf.Tensor)] = {
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
      npath -> conv.tensor2()
    }
    elts
  }


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
      tv: Seq[tf.Tensor],
      tf_struct: StructType,
      input: Array[Row],
      input_struct: StructType,
      appendInput: Boolean): Iterator[Row] = {
    // The structures should already have been validated.
    // Output has all the TF columns first, and then the other columns
    logDebug(s"convertBack: ${input.length} input rows, tv=$tv tf_struct=$tf_struct input_struct=$input_struct " +
      s"append=$appendInput")

    val tfSizesAndIters = for ((field, t) <- tf_struct.fields.zip(tv).toSeq) yield {
      val info = ColumnInformation(field).stf.getOrElse {
        throw new Exception(s"Missing info in field $field")
      }
//      logTrace(s"convertBack: $field $info")
      // Drop the first cell, this is a block.
      val expLength = if (appendInput) { Some(input.length) } else { None }
      val (numRows, iter) = getColumn(t, info.dataType, info.shape.tail, expLength)
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
    val res: Iterator[Row] = DataOps.convertBackFast0(input, tfIters, tfNumRows, input_struct, outputSchema)
    res
  }


  /**
    * Extracts the content of a column as objects amenable to SQL.
    *
    * @param t
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
      t: tf.Tensor,
      scalaType: ScalarType,
      cellShape: Shape,
      expectedNumRows: Option[Int],
      fastPath: Boolean = true): (Int, Iterable[Any]) = {
    val allDataBuffer: mutable.WrappedArray[_] =
      SupportedOperations.opsFor(scalaType).convertTensor(t)
    val numData = allDataBuffer.size
    // Infer if necessary the reshaping size.
    val (inferredNumRows, inferredShape) = DataOps.inferPhysicalShape(numData, cellShape, expectedNumRows)
    val reshapeShape = inferredShape.prepend(inferredNumRows)
    val res = if (fastPath) {
      DataOps.getColumnFast0(reshapeShape, scalaType, allDataBuffer)
    } else {
      DataOps.reshapeIter(allDataBuffer.asInstanceOf[mutable.WrappedArray[Any]],
        inferredShape.dims.toList)
    }
    inferredNumRows -> res
  }

}
