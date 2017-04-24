package org.tensorframes

import scala.reflect.runtime.universe.TypeTag
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{IntegerType, NumericType}
import org.tensorframes.impl.SupportedOperations

/**
 * The public interface that reimplements a subset of the official TensorFlow python API.
 *
 * The name of the functions and of the parameters should be the same as in python.
 *
 * Note that elementary operations such as +, -, etc. can be directly expressed using the
 * standard mathematical operators.
 */
package object dsl {

  import DslImpl._

  private[dsl] implicit def op2Node(op: Operation): Node = op.asInstanceOf[Node]
  private[dsl] implicit def opSeq2NodeSeq(op: Seq[Operation]): Seq[Node] = {
    op.map(op2Node)
  }

  private[dsl] implicit class RichOperation(op: Operation) {
    def n: Node = op2Node(op)
  }

  // ******* Control flow *********

  def scope[T](pathElem: String)(fun: => T): T = {
    Paths.withScope(pathElem)(fun)
  }

  def withGraph[T](fun: => T): T = Paths.withGraph(fun)

  /**
   * Used to express unknown dimensions.
   */
  val Unknown = Shape.Unknown

  // ******** Constants, Sequences, and Random Values ***********

  def placeholder[T : Numeric : TypeTag](shape: Int*): Operation = {
    val ops = SupportedOperations.getOps[T]()
    DslImpl.placeholder(ops.sqlType.asInstanceOf[NumericType], Shape(shape: _*))
  }

  def constant[T : ConvertibleToDenseTensor](x: T): Operation = {
    val ev = implicitly[ConvertibleToDenseTensor[T]]
    build_constant(ev.tensor(x))
  }

  def zeros(shape: Int*): Operation = zeros[Float](shape:_*)

  def zeros[T : Numeric : ConvertibleToDenseTensor : TypeTag](shape: Int*): Operation = {
    fill(shape, implicitly[Numeric[T]].zero)
  }

  def ones(shape: Int*): Operation = ones[Float](shape: _*)

  def ones[T : Numeric : ConvertibleToDenseTensor : TypeTag](shape: Int*): Operation = {
    fill(shape, implicitly[Numeric[T]].one)
  }

  def fill[T : Numeric : ConvertibleToDenseTensor : TypeTag](
      dims: Seq[Int], value: T): Operation = {
    dims match {
      case Seq() =>
        fill(constant(1), constant(value))
      case Seq(n) =>
        fill(constant(Seq(n)), constant(value))
      case _ =>
        throw HighDimException(Shape(dims: _*))
    }
  }

  def fill(dims: Operation, value: Operation): Operation = {
    require(dims.n.scalarType == IntegerType)
    require(value.n.shape.numDims == 0, value.n.shape)
    build("Fill",
      shape = dims.n.shape,
      dtype = value.n.scalarType,
      extraParents = (p: String) => Seq(dims.named(p + "/dims"), value.named(p + "/value")))
  }

  /**
   * Builds a block placeholder based on the content of a column in a dataframe.
   *
   * @param df a dataframe
   * @param colName the name of a column in a dataframe
   * @return a placeholder
   */
  def block(df: DataFrame, colName: String, tfName: String): Operation = {
    extractPlaceholder(df, colName, tfName, block = true)
  }

  /**
   * Builds a row placeholder based on the content of a column in a dataframe.
   */
  def row(df: DataFrame, colName: String, tfName: String): Operation = {
    extractPlaceholder(df, colName, tfName, block = false)
  }

  // ******** Tensor Transformations ***********


  def identity(op: Operation): Operation = {
    build("Identity", parents = Seq(op))
  }

  def add(x: Operation, y: Operation): Operation =
    build("Add", parents=Seq(x, y), shapeInfer = broadcastShape)

  def div(x: Operation, y: Operation): Operation =
    build("Div", parents=Seq(x, y), shapeInfer = broadcastShape)

  // **** Reducers ******

  def reduce_min(
      input_tensor: Operation,
      reduction_indices: Seq[Int] = null): Operation = {
    build_reducer("Min", input_tensor, reduction_indices)
  }

  def reduce_sum(
      input_tensor: Operation,
      reduction_indices: Seq[Int] = null): Operation = {
    build_reducer("Sum", input_tensor, reduction_indices)
  }

}

