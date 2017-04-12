package org.tensorframes.impl

import scala.collection.mutable
import scala.reflect.ClassTag

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types.{NumericType, StructType}

import org.tensorframes.{Logging, Shape}
import org.tensorframes.Shape.DimType

/**
  * Converts data between the C++ runtime of TensorFlow and the Spark runtime.
 *
 * Current (only) implementation exports each row and copies it back into a C++ buffer.
 */
object DataOps extends Logging {

  def convertBackFast0(
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

  def convertFast0(
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

  // **** Conversions TF => Spark ***

  def reshapeIter[T : ClassTag](
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
  def inferPhysicalShape(
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

  def getColumnFast0(
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
}
