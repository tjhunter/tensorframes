package org.tensorframes.test

import java.nio.file.{Files, Paths}

import org.apache.spark.sql.types.{DataType, NumericType}
import org.tensorflow.framework.{AttrValue, GraphDef, NodeDef, TensorShapeProto, DataType => ProtoDataType}
import org.tensorframes.{Logging, Shape}
import org.tensorframes.impl.{DenseTensor, ScalarType, SupportedOperations}

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe._

/**
 * Some testing utilities into what may become a DSL at some point.
 */
object dsl extends Logging {

  import ProtoConversions._

  private implicit class ShapeToAttr(s: Shape) {
    def toAttr: AttrValue = AttrValue.newBuilder().setShape(buildShape(s)).build()
  }

  private implicit class SQLTypeToAttr(s: NumericType) {
    def toAttr: AttrValue = buildType(s)
  }

  private implicit class DataTypeToAttr(dt: ProtoDataType) {
    def toAttr: AttrValue = dataTypeToAttrValue(dt)
  }

  implicit class RichDouble(x: Double) {
    def +(n: Node): Node = {
      op_add(constant(x), n)
    }
  }

  private var counter: Int = 0

  case class Node(
      name: String,
      opName: String,
      scalarType: NumericType,
      shape: Shape,
      parents: Seq[Node],
      isOp: Boolean,
      extraAttr: Map[String, AttrValue]) {

    def node: NodeDef = {
      val b = NodeDef.newBuilder()
      b.setName(name)
      b.setOp(opName)
      parents.foreach(p => b.addInput(p.name))
      if (isOp) {
        b.putAllAttr(Map("T" -> getDType(scalarType).toAttr).asJava)
      } else {
        b.putAllAttr(Map(
          "dtype" -> getDType(scalarType).toAttr).asJava)
      }
      b.putAllAttr(extraAttr.asJava)
      b.build()
    }

    def named (newName: String): Node = copy(name = newName)

    def +(other: Node): Node = op_add(this, other)
  }

  private[tensorframes] def placeholder(dtype: DataType, shape: Shape): Node = {
    build("Placeholder", shape=shape, dtype=dtype.asInstanceOf[NumericType], isOp = false,
      extraAttrs = Map("shape" -> shape.toAttr))
  }

  def placeholder[T : Numeric : TypeTag](shape: Int*): Node = {
    val ops = SupportedOperations.getOps[T]()
    placeholder(ops.sqlType, Shape(shape: _*))
  }

  def constant[T : Numeric : TypeTag](x: T): Node = {
    build_constant(DenseTensor(x))
  }

  def constant[T : Numeric : TypeTag](x: Seq[T]): Node = {
    build_constant(DenseTensor(x))
  }

  def op_id(node: Node): Node = {
    build("Identity", parents = Seq(node))
  }

  def op_add(node1: Node, node2: Node): Node = {
    build("Add", parents=Seq(node1, node2))
  }

  // Common loading and building

  def buildGraph(nodes: Node*): GraphDef = {
    logDebug(s"buildGraph for nodes: ${nodes.map(_.name)}")
    var treated: Map[String, Node] = Map.empty
    nodes.foreach { n =>
      logDebug(s"call: n=${n.name}, treated=${treated.keySet}")
      treated = getClosure(n, treated)
    }
    val b = GraphDef.newBuilder()
    treated.values.map(_.node).foreach(b.addNode)
    b.build()
  }

  def load(file: String): GraphDef = {
    val byteArray = Files.readAllBytes(Paths.get(file))
    GraphDef.newBuilder().mergeFrom(byteArray).build()
  }

  private def commonShape(shapes: Seq[Shape]): Shape = {
    require(shapes.nonEmpty)
    require(shapes.forall(_ == shapes.head), s"$shapes")
    shapes.head
  }

  private def commonType(dtypes: Seq[NumericType]): NumericType = {
    require(dtypes.nonEmpty)
    require(dtypes.forall(_ == dtypes.head))
    dtypes.head
  }

  private def build(
      opName: String,
      name: String = null,
      parents: Seq[Node] = Seq.empty,
      isOp: Boolean = true,
      dtype: NumericType = null,
      shape: Shape = null,
      dtypeInfer: Seq[NumericType] => NumericType = commonType,
      shapeInfer: Seq[Shape] => Shape = commonShape,
      extraAttrs: Map[String, AttrValue] = Map.empty): Node = {
    val n = Option(name).getOrElse {
      counter += 1
      s"${opName}_$counter"
    }
    val dt = Option(dtype).getOrElse(dtypeInfer(parents.map(_.scalarType)))
    val sh = Option(shape).getOrElse(shapeInfer(parents.map(_.shape)))
    Node(n, opName, dt, sh, parents, isOp, extraAttrs)
  }

  private def buildShape(s: Shape): TensorShapeProto = s.toProto

  private def buildType(sqlType: NumericType): AttrValue = {
    AttrValue.newBuilder().setType(getDType(sqlType)).build()
  }

  private def uniqueByName(nodes: Seq[Node]): Map[String, Node] = {
    nodes.groupBy(_.name).mapValues(_.head)
  }

  private def getClosure(node: Node, treated: Map[String, Node]): Map[String, Node] = {
    logDebug(s"closure: n=${node.name}, parents=${node.parents.map(_.name)}," +
      s" treated=${treated.keySet}")
    val explored = node.parents
      .filterNot(n => treated.contains(n.node.getName))
      .flatMap(getClosure(_, treated + (node.name -> node)))
      .toMap

    uniqueByName(node +: (explored.values.toSeq ++ treated.values.toSeq))
  }

  private def build_constant(dt: DenseTensor): Node = {
    val a = AttrValue.newBuilder().setTensor(DenseTensor.toTensorProto(dt))
    val dt2 = SupportedOperations.opsFor(dt.dtype).sqlType.asInstanceOf[NumericType]
    build("Const", isOp = false,
      shape = dt.shape, dtype = dt2,
      extraAttrs = Map("value" -> a.build()))
  }

  // Reducers (unfinished business)

  def reduce_min(
      input_tensor: Node,
      reduction_indices: Seq[Int] = null,
      name: String = null): Node = {
    build_reducer("Min", input_tensor, reduction_indices, name)
  }

  def reduce_sum(
      input_tensor: Node,
      reduction_indices: Seq[Int] = null,
      name: String = null): Node = {
    build_reducer("Sum", input_tensor, reduction_indices, name)
  }

  private def build_reducer(
      opName: String,
      parent: Node,
      reduction_indices: Seq[Int] = null,
      name: String = null): Node = {
    val idxs = constant(reduction_indices) named (parent.name + "/reduction_indices")
    build(opName, name, Seq(parent, idxs),
      dtype = parent.scalarType,
      shape = reduce_shape(parent.shape, Option(reduction_indices).getOrElse(Nil)),
      extraAttrs = Map(
        "Tidx" -> AttrValue.newBuilder().setType(ProtoDataType.DT_INT32).build(),
        "keep_dims" -> AttrValue.newBuilder().setB(false).build()))
  }

  private def reduce_shape(s: Shape, red_indices: Seq[Int]): Shape = {
    require(s.numDims >= red_indices.size)
    // Special case for empty
    if (red_indices.isEmpty) {
      Shape.empty
    } else {
      // The remaining dimensions:
      val rem = s.dims.indices.filterNot(red_indices.contains)
      Shape(rem: _*)
    }
  }

}

/**
 * Utilities to convert data back and forth between the proto descriptions and the dataframe descriptions.
 */
object ProtoConversions {
  def getDType(nodeDef: NodeDef): ProtoDataType = {
    val opt = Option(nodeDef.getAttr.get("T")).orElse(Option(nodeDef.getAttr.get("dtype")))
    val v = opt.getOrElse(throw new Exception(s"Neither 'T' no 'dtype' was found in $nodeDef"))
    v.getType
  }

  def getDType(sqlType: NumericType): ProtoDataType = {
    SupportedOperations.opsFor(sqlType).tfType
  }
  def getDType(sqlType: ScalarType): ProtoDataType = {
    SupportedOperations.opsFor(sqlType).tfType
  }

  def sqlTypeToAttrValue(sqlType: NumericType): AttrValue = {
    AttrValue.newBuilder().setType(getDType(sqlType)).build()
  }

  def dataTypeToAttrValue(dataType: ProtoDataType): AttrValue = {
    AttrValue.newBuilder().setType(dataType).build()
  }

}
