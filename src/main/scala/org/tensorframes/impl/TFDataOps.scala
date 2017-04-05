package org.tensorframes.impl

import scala.collection.mutable
import scala.reflect.ClassTag
import org.{tensorflow => tf}

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.{GenericRow, GenericRowWithSchema}
import org.apache.spark.sql.types.{NumericType, StructType}
import org.tensorframes.{ColumnInformation, Logging, NodePath, Shape}
import org.tensorframes.Shape.DimType

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
      requestedTFCols: Array[Int]): Seq[(String, tf.Tensor)] = {
    // This is a very simple and very inefficient implementation. It should be kept
    // as is for correctness checks.
    val fields = requestedTFCols.map(struct.fields(_))
    logDebug(s"convert2: Calling convert on ${it.length} rows with struct: $struct " +
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

    convertFast0(it, converters, requestedTFCols)

    val tensors = converters.map(_.tensor2())
    val names = requestedTFCols.map(struct(_).name)
    names.zip(tensors)
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

}
