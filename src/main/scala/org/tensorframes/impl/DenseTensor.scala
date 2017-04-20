package org.tensorframes.impl

import java.nio.{ByteBuffer, ByteOrder}

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe._
import org.tensorflow.framework.TensorProto
import org.tensorframes.Shape
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, NumericType}

/**
 * Binary representation of tensorial data.
 *
 * This class is not meant to be used in computation. It is a placeholder to efficiently encode
 * all of TensorFlow's dense data (using directly protocol buffers has performance issues, it
 * seems).
 */
private[tensorframes] class DenseTensor private(
    val shape: Shape,
    val dtype: NumericType,
    private val data: Array[Byte]) {

  override def toString(): String = s"DenseTensor($shape, $dtype, " +
    s"${data.length / dtype.defaultSize} elements)"
}

private[tensorframes] object DenseTensor {
  def apply[T](x: T)(implicit ev2: TypeTag[T]): DenseTensor = {
    val ops = SupportedOperations.getOps[T]()
    new DenseTensor(Shape.empty, ops.sqlType, convert(x))
  }

  def apply[T](xs: Seq[T])(implicit ev1: Numeric[T], ev2: TypeTag[T]): DenseTensor = {
    val ops = SupportedOperations.getOps[T]()
    new DenseTensor(Shape(xs.size), ops.sqlType, convert1(xs))
  }

  def matrix[T](xs: Seq[Seq[T]])(implicit ev1: Numeric[T], ev2: TypeTag[T]): DenseTensor = {
    val ops = SupportedOperations.getOps[T]()
    new DenseTensor(Shape(xs.size, xs.head.size), ops.sqlType, convert2(xs))
  }

  private def convert[T](x: T)(implicit ev2: TypeTag[T]): Array[Byte] = {
    val ops = SupportedOperations.getOps[T]()
    val conv = ops.tfConverter(Shape.empty, 1)
    conv.reserve()
    conv.appendRaw(x)
    conv.toByteArray()
  }

  private def convert1[T](xs: Seq[T])(implicit ev1: Numeric[T], ev2: TypeTag[T]): Array[Byte] = {
    val ops = SupportedOperations.getOps[T]()
    val conv = ops.tfConverter(Shape.empty, xs.size)
    conv.reserve()
    xs.foreach(conv.appendRaw)
    conv.toByteArray()
  }

  private def convert2[T](xs: Seq[Seq[T]])(
      implicit ev1: Numeric[T], ev2: TypeTag[T]): Array[Byte] = {
    val ops = SupportedOperations.getOps[T]()
    val conv = ops.tfConverter(Shape.empty, xs.size * xs.head.size)
    conv.reserve()
    xs.foreach(_.foreach(conv.appendRaw))
    conv.toByteArray()
  }

  // These operations need to discriminate between types, they should be moved to datatypeops.
  def toTensorProto(t: DenseTensor): TensorProto = {
    val b = TensorProto.newBuilder()
    val ops = SupportedOperations.opsFor(t.dtype)
    b.setTensorShape(t.shape.toProto)
    b.setDtype(ops.tfType)
    // Watch out for the bit order. It seems that wrapping does not use the same ordering.
    val rawBuffer = ByteBuffer.wrap(t.data).order(ByteOrder.LITTLE_ENDIAN)
    ops.sqlType match {
      case DoubleType =>
        val buff = rawBuffer.asDoubleBuffer()
        buff.rewind()
        (0 until buff.limit()).foreach(i => b.addDoubleVal(buff.get(i)))
      case FloatType =>
        val buff = rawBuffer.asFloatBuffer()
        buff.rewind()
        (0 until buff.limit()).foreach(i => b.addFloatVal(buff.get(i)))
      case IntegerType =>
        val buff = rawBuffer.asIntBuffer()
        buff.rewind()
        (0 until buff.limit()).foreach(i => b.addIntVal(buff.get(i)))
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot convert type ${ops.sqlType}")
    }
    b.build()
  }

  def apply(proto: TensorProto): DenseTensor = {
    val ops = SupportedOperations.opsFor(proto.getDtype)
    val shape = Shape.from(proto.getTensorShape)
    val data = ops.sqlType match {
      case DoubleType =>
        val coll = proto.getDoubleValList.asScala.toSeq.map(_.doubleValue())
        convert(coll)
      case IntegerType =>
        val coll = proto.getIntValList.asScala.toSeq.map(_.intValue())
        convert(coll)
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot convert type ${ops.sqlType}")
    }
    new DenseTensor(shape, ops.sqlType, data)
  }
}
