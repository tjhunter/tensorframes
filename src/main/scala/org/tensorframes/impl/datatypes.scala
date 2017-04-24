package org.tensorframes.impl

import java.nio._

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.{tensorflow => tf}
import org.tensorflow.framework.{DataType => ProtoDataType}
import org.tensorframes.{Logging, Shape}

import scala.collection.mutable.{WrappedArray => MWrappedArray}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

// All the datatypes supported by TensorFrames
// TODO: all the buffer operations should specify their byte orders:
//  - TensorFlow: ???
//  - jvm: ???
//  - protobuf: ???

/**
 * All the types of scalars supported by TensorFrames.
 *
 * It can be argued that the Binary type is not really a scalar,
 * but it is considered as such by both Spark and TensorFlow.
 */
trait ScalarType

/**
 * Int32
 */
case object ScalarIntType extends ScalarType

/**
 * INT64
 */
case object ScalarLongType extends ScalarType

/**
 * FLOAT64
 */
case object ScalarDoubleType extends ScalarType

/**
 * FLOAT32
 */
case object ScalarFloatType extends ScalarType

/**
 * STRING / BINARY
 */
case object ScalarBinaryType extends ScalarType

/**
 * @param shape the shape of the element in the row (not the overall shape of the block)
 * @param numCells the number of cells that are going to be allocated with the given shape.
 * @tparam T
 */
// TODO PERF: investigate the use of @specialized
private[tensorframes] sealed abstract class TensorConverter[@specialized(Double, Float, Int, Long) T] (
    val shape: Shape,
    val numCells: Int)
  (implicit ev2: ClassTag[T]) extends Logging {
  final val empty = Array.empty[T]
  /**
   * Creates memory space for a given number of units of the given shape.
   *
   * Can only be called once.
   */
  def reserve(): Unit

  def appendRaw(t: T): Unit

  protected lazy val numElements: Int = {
    val numElts = fullShape.numElements.get.toInt
    assert(numElts < Int.MaxValue, s"Cannot reserve $numElts (max allowed=${Int.MaxValue}")
    numElts.toInt
  }

  protected lazy val fullShape: Shape = {
    assert(! shape.hasUnknown, s"Shape $shape has unknown values.")
    shape.prepend(numCells)
  }

  private[impl] final def appendArray(arr: MWrappedArray[T]): Unit = {
    var idx = 0
    while (idx < arr.length) {
      appendRaw(arr(idx))
      idx += 1
    }
  }

  // The data returned in row objects is boxed, there is not much we can do here...
  private[impl] final def appendSeq(data: Seq[T]): Unit = {
    data match {
      case wa : MWrappedArray[T @unchecked] =>
        // Faster path
        appendArray(wa)
      case seq: Any =>
        throw new Exception(s"Type ${seq.getClass} is a slow path")
    }
  }

  private[impl] final def appendSeqSeq(data: Seq[Seq[T]]): Unit = {
    data match {
      case wa : MWrappedArray[Seq[T] @unchecked] =>
        wa.foreach(appendSeq)
      case seq: Any =>
        throw new Exception(s"Type ${seq.getClass} is a slow path")
    }
  }

  // The return element is just here so that the method gets specialized (otherwise it would not).
  final def append(row: Row, position: Int): Array[T] = {
    logger.debug(s"append: position=$position row=$row")
    val d = shape.numDims
    if (d == 0) {
      appendRaw(row.getAs[T](position))
    } else if (d == 1) {
      appendSeq(row.getSeq[T](position))
    } else if (d == 2) {
      appendSeqSeq(row.getSeq[Seq[T]](position))
    } else if (d > 2 || d < 0) {
      throw new Exception(s"Higher order dimension $d from shape $shape is not supported")
    }
    empty
  }

  def tensor2(): tf.Tensor

  /**
   * The physical size of a single element, in bytes.
   */
  protected val elementSize: Int = -1

  /**
   * Fills a buffer (which has been previously allocated and reset) with the content of the
   * current tensor.
   */
  protected def fillBuffer(buff: ByteBuffer): Unit

  def toByteArray(): Array[Byte] = {
    val array = Array.ofDim[Byte](numElements * elementSize)
    // Watch out for the endianness. This is important in order to respect the order in
    // the protos.
    // Not sure if this causes some performance issues in the TensorFlow side or the Spark side.
    val b = ByteBuffer.wrap(array).order(ByteOrder.LITTLE_ENDIAN)
    b.rewind()
    fillBuffer(b)
    array
  }
}


/**
 * All the operations required to support a given type.
 *
 * It assumes that there is a one-to-one correspondence between SQL types, scala types and TF types.
 * It does not support TF's rich type collection (uint16, float128, etc.). These have to be handled
 * internally through casting.
 */
private[tensorframes] sealed abstract class ScalarTypeOperation[@specialized(Int, Long, Double, Float) T] {
  /**
   * The SQL type associated with the given type.
   */
  val sqlType: DataType

  /**
   * The TF type
   */
  val tfType: ProtoDataType

  /**
   * The TF type (new style).
   *
   */
  val tfType2: tf.DataType

  /**
   * The type of the scalar value.
   */
  val scalarType: ScalarType

  /**
   * A zero element for this type
   */
  val zero: T

  /**
   * Returns a new converter for Array[Row] -> DenseTensor
   */
  def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[T]

  /**
   * The conversion of DenseTensor -> Row stuff.
   *
   * @param numElements the number of elements expected to be found in the tensor.
   */
  @deprecated("to remove", "now")
  def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any]

  def convertTensor(t: tf.Tensor): MWrappedArray[T]

  final def convertBuffer1(b0: Array[_], dim1: Int): Array[T] = {
    val b = b0.asInstanceOf[Array[T]]
    assert(b.length == dim1, (b.length, dim1))
    b
  }

  final def convertBuffer2(b0: Array[_], dim1: Int, dim2: Int): Array[MWrappedArray[T]] = {
    val b = b0.asInstanceOf[Array[T]]
    assert(b.length == dim1 * dim2, (b.length, dim1, dim2))
    val res = Array.fill(dim1){ new Array[T](dim2) }
    var idx1 = 0
    while (idx1 < dim1) {
      var idx2 = 0
      while(idx2 < dim2) {
        // TODO(tjh) check math
        res(idx1)(idx2) = b(idx2 + dim2 * idx1)
        idx2 += 1
      }
      idx1 += 1
    }
    res.map(conv)
  }

  // Necessary hack to have proper scala sequences. Spark's Row object does not like basic arrays.
  private def conv[X](m: Array[X]): MWrappedArray[X] = {
    m.toSeq.asInstanceOf[MWrappedArray[X]]
  }

  final def convertBuffer3(
      b0: Array[_],
      dim1: Int,
      dim2: Int,
      dim3: Int): Array[MWrappedArray[MWrappedArray[T]]] = {
    val b = b0.asInstanceOf[Array[T]]
    assert(b.length == dim1 * dim2 * dim3, (b.length, dim1, dim2, dim3))
    val res = Array.fill(dim1) { Array.fill(dim2) { new Array[T](dim3) } }
    var idx1 = 0
    while (idx1 < dim1) {
      var idx2 = 0
      while(idx2 < dim2) {
        var idx3 = 0
        while (idx3 < dim3) {
          // TODO(tjh) check math
          res(idx1)(idx2)(idx3) = b(idx3 + dim3 * idx2 + dim3 * dim2 * idx1)
          idx3 += 1
        }
        idx2 += 1
      }
      idx1 += 1
    }
    // Because rows cannot deal with primitive arrays, some weird conversions need to be done.
    res.map { arr => conv(arr.map(conv)) }
  }

  implicit def classTag: ClassTag[T] = ev

  def tag: Option[TypeTag[_]]

  def ev: ClassTag[T] = null
}

private[tensorframes] object SupportedOperations {
  private val ops: Seq[ScalarTypeOperation[_]] =
    Seq(DoubleOperations, FloatOperations, IntOperations, LongOperations, StringOperations)

  val sqlTypes = ops.map(_.sqlType)

  val scalarTypes = ops.map(_.scalarType)

  private val tfTypes = ops.map(_.tfType)

  def getOps(t: DataType): Option[ScalarTypeOperation[_]] = {
    ops.find(_.sqlType == t)
  }

  def opsFor(t: DataType): ScalarTypeOperation[_] = {
    ops.find(_.sqlType == t).getOrElse {
      throw new IllegalArgumentException(s"Type $t is not supported. Only the following types are" +
        s"supported: ${sqlTypes.mkString(", ")}")
    }
  }

  def opsFor(t: ScalarType): ScalarTypeOperation[_] = {
    ops.find(_.scalarType == t).getOrElse {
      throw new IllegalArgumentException(s"Type $t is not supported. Only the following types are" +
        s"supported: ${sqlTypes.mkString(", ")}")
    }
  }

  def opsFor(t: ProtoDataType): ScalarTypeOperation[_] = {
    ops.find(_.tfType == t).getOrElse {
      throw new IllegalArgumentException(s"Type $t is not supported. Only the following types are" +
        s"supported: ${tfTypes.mkString(", ")}")
    }
  }

  def opsFor(dt: tf.DataType): ScalarTypeOperation[_] = {
    ops.find(_.tfType2 == dt).getOrElse {
      throw new IllegalArgumentException(s"Type $dt is not supported. Only the following types are" +
        s"supported: ${tfTypes.mkString(", ")}")
    }

  }

  def getOps[T : TypeTag](): ScalarTypeOperation[T] = {
    val ev: TypeTag[_] = implicitly[TypeTag[T]]
    ops.find(_.tag.map(_.tpe =:= ev.tpe) == Some(true)).getOrElse {
      val tags = ops.map(_.tag.toString()).mkString(", ")
      throw new IllegalArgumentException(s"Type ${ev} is not supported. Only the following types " +
        s"are supported: ${tags}")
    }   .asInstanceOf[ScalarTypeOperation[T]]
  }

  def hasOps(x: Any): Boolean = x match {
    case _: Double => true
    case _: Int => true
    case _: Long => true
    case _: Float => true
    case _ => false
  }
}

// ********** DOUBLES ************

private[impl] class DoubleTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Double](s, numCells) with Logging {
  private var buffer: DoubleBuffer = null

  override val elementSize: Int = 8

  override def reserve(): Unit = {
    logTrace(s"Reserving for $numCells units of shape $shape")
    buffer = DoubleBuffer.allocate(numElements)
    buffer.rewind()
  }

  override def appendRaw(d: Double): Unit = {
    buffer.put(d)
  }

  override def tensor2(): tf.Tensor = {
    buffer.rewind()
    tf.Tensor.create(fullShape.dims.toArray, buffer)
  }

  override def fillBuffer(buff: ByteBuffer): Unit = {
    buffer.rewind()
    buff.asDoubleBuffer().put(buffer)
  }
}

private[impl] object DoubleOperations extends ScalarTypeOperation[Double] with Logging {
  override val sqlType = DoubleType
  override val tfType = ProtoDataType.DT_DOUBLE
  override val tfType2 = tf.DataType.DOUBLE
  override val scalarType = ScalarDoubleType
  final override val zero = 0.0
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Double] =
    new DoubleTensorConverter(cellShape, numCells)


  override def convertTensor(t: tf.Tensor): MWrappedArray[Double] = {
    val res: Array[Double] = Array.fill(t.numElements())(Double.NaN)
    val b = DoubleBuffer.wrap(res)
    t.writeTo(b)
    res
  }

  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asDoubleBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    logTrace(s"convertBuffer: dbuff: pos:${dbuff.position()}, cap:${dbuff.capacity()} " +
      s"limit:${dbuff.limit()} expected=$numElements")
    val res: Array[Double] = Array.fill(numBufferElements)(Double.NaN)
    dbuff.get(res)
    logTrace(s"Extracted from buffer: ${res.toSeq}")
    res
  }

  override def tag: Option[TypeTag[_]] = Option(implicitly[TypeTag[Double]])

  override def ev = ClassTag.Double
}

// ********** FLOAT ************

private[impl] class FloatTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Float](s, numCells) with Logging {
  private var buffer: FloatBuffer = null

  override val elementSize: Int = 4

  override def reserve(): Unit = {
    logTrace(s"Reserving for $numCells units of shape $shape")
    buffer = FloatBuffer.allocate(numElements)
    buffer.rewind()
  }

  override def appendRaw(d: Float): Unit = {
    buffer.put(d)
  }

  override def tensor2(): tf.Tensor = {
    buffer.rewind()
    tf.Tensor.create(fullShape.dims.toArray, buffer)
  }

  override def fillBuffer(buff: ByteBuffer): Unit = {
    buffer.rewind()
    buff.asFloatBuffer().put(buffer)
  }
}

private[impl] object FloatOperations extends ScalarTypeOperation[Float] with Logging {
  override val sqlType = FloatType
  override val tfType = ProtoDataType.DT_FLOAT
  override val tfType2 = tf.DataType.FLOAT
  override val scalarType = ScalarFloatType
  final override val zero = 0.0f
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Float] =
    new FloatTensorConverter(cellShape, numCells)
  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asFloatBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    logTrace(s"convertBuffer: dbuff: pos:${dbuff.position()}, cap:${dbuff.capacity()} " +
      s"limit:${dbuff.limit()} expected=$numElements")
    val res: Array[Float] = Array.fill(numBufferElements)(Float.NaN)
    dbuff.get(res)
    logTrace(s"Extracted from buffer: ${res.toSeq}")
    res
  }

  override def convertTensor(t: tf.Tensor): MWrappedArray[Float] = {
    val res: Array[Float] = Array.fill(t.numElements())(Float.NaN)
    val b = FloatBuffer.wrap(res)
    t.writeTo(b)
    res
  }

  override def tag: Option[TypeTag[_]] = Option(implicitly[TypeTag[Float]])

  override def ev = ClassTag.Float
}

// ********** INT32 ************

private[impl] class IntTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Int](s, numCells) with Logging {
  private var buffer: IntBuffer = null

  override val elementSize: Int = 4

  override def reserve(): Unit = {
    logTrace(s"Reserving for $numCells units of shape $shape")
    buffer = IntBuffer.allocate(numElements)
    buffer.rewind()
  }

  override def appendRaw(d: Int): Unit = {
    buffer.put(d)
  }

  override def tensor2(): tf.Tensor = {
    buffer.rewind()
    tf.Tensor.create(fullShape.dims.toArray, buffer)
  }

  override def fillBuffer(buff: ByteBuffer): Unit = {
    buffer.rewind()
    buff.asIntBuffer().put(buffer)
  }
}

private[impl] object IntOperations extends ScalarTypeOperation[Int] with Logging {
  override val sqlType = IntegerType
  override val tfType = ProtoDataType.DT_INT32
  override val tfType2 = tf.DataType.INT32
  override val scalarType = ScalarIntType
  final override val zero = 0
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Int] =
    new IntTensorConverter(cellShape, numCells)

  override def convertTensor(t: tf.Tensor): MWrappedArray[Int] = {
    val res: Array[Int] = Array.fill(t.numElements())(0)
    val b = IntBuffer.wrap(res)
    t.writeTo(b)
    res
  }

  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asIntBuffer()
    dbuff.rewind()
    val res: Array[Int] = Array.fill(numElements)(Int.MinValue)
    dbuff.get(res)
    res
  }

  override def tag: Option[TypeTag[_]] = Option(implicitly[TypeTag[Int]])

  override def ev = ClassTag.Int
}

// ****** INT64 (LONG) ******

private[impl] class LongTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Long](s, numCells) with Logging {
  private var buffer: LongBuffer = null

  override val elementSize: Int = 8

  override def reserve(): Unit = {
    logTrace(s"Reserving for $numCells units of shape $shape")
    buffer = LongBuffer.allocate(numElements)
    buffer.rewind()
  }

  override def appendRaw(d: Long): Unit = {
    buffer.put(d)
  }

  override def tensor2(): tf.Tensor = {
    buffer.rewind()
    tf.Tensor.create(fullShape.dims.toArray, buffer)
  }

  override def fillBuffer(buff: ByteBuffer): Unit = {
    buffer.rewind()
    buff.asLongBuffer().put(buffer)
  }
}

private[impl] object LongOperations extends ScalarTypeOperation[Long] with Logging {
  override val sqlType = LongType
  override val tfType = ProtoDataType.DT_INT64
  override val tfType2 = tf.DataType.INT64
  override val scalarType = ScalarLongType
  final override val zero = 0L
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Long] =
    new LongTensorConverter(cellShape, numCells)

  override def convertTensor(t: tf.Tensor): MWrappedArray[Long] = {
    val res: Array[Long] = Array.fill(t.numElements())(0L)
    val b = LongBuffer.wrap(res)
    t.writeTo(b)
    res
  }

  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asLongBuffer()
    dbuff.rewind()
    val res: Array[Long] = Array.fill(numElements)(Long.MinValue)
    dbuff.get(res)
    logTrace(s"Extracted from buffer: ${res.toSeq}")
    res
  }

  override def tag: Option[TypeTag[_]] = Option(implicitly[TypeTag[Long]])

  override def ev = ClassTag.Long
}

// ********** STRING *********
// This is actually byte arrays, which corresponds to the 'binary' type in Spark.

// The string converter can only deal with one row at a time (the most common case).
private[impl] class StringTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Array[Byte]](s, numCells) with Logging {
  private var buffer: Array[Byte] = null

  override val elementSize: Int = 1

  {
    logger.debug(s"Creating string buffer for shape $s and $numCells cells")
    assert(s == Shape() && numCells == 1, s"The string buffer does not accept more than one" +
      s" scalar of type binary. shape=$s numCells=$numCells")
  }


  override def reserve(): Unit = {}

  override def appendRaw(d: Array[Byte]): Unit = {
    assert(buffer == null, s"The buffer has only been set with ${buffer.length} values," +
      s" but ${d.length} are trying to get inserted")
    buffer = d.clone()
  }

  override def tensor2(): tf.Tensor = {
    tf.Tensor.create(buffer)
  }

  override def fillBuffer(buff: ByteBuffer): Unit = {
    buff.put(buffer)
  }
}

private[impl] object StringOperations extends ScalarTypeOperation[Array[Byte]] with Logging {
  override val sqlType = BinaryType
  override val tfType = ProtoDataType.DT_STRING
  override val tfType2 = tf.DataType.STRING
  override val scalarType = ScalarBinaryType
  final override val zero = Array.empty[Byte]

  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Array[Byte]] =
    new StringTensorConverter(cellShape, numCells)

  override def convertTensor(t: tf.Tensor): MWrappedArray[Array[Byte]] = {
    throw new Exception(s"convertTensor is not implemented for strings")
  }

  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    throw new Exception(s"convertBuffer is not implemented for strings")
  }

  override def tag: Option[TypeTag[_]] = None

  override def ev = throw new Exception(s"ev is not implemented for strings")
}


