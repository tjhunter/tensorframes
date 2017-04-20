package org.tensorframes.dsl

import javax.annotation.Nullable
import org.tensorflow.framework.{AttrValue, DataType, GraphDef, TensorShapeProto}

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.NumericType

import org.tensorframes.{Logging, ColumnInformation, Shape}
import org.tensorframes.impl.DenseTensor


/**
 * Implementation of the DSL operations.
 */
private[dsl] object DslImpl extends Logging with DefaultConversions {
  import ProtoConversions._

  private[dsl] implicit class ShapeToAttr(s: Shape) {
    def toAttr: AttrValue = AttrValue.newBuilder().setShape(buildShape(s)).build()
  }

  private[dsl] implicit class SQLTypeToAttr(s: NumericType) {
    def toAttr: AttrValue = buildType(s)
  }

  private[dsl] implicit class DataTypeToAttr(dt: DataType) {
    def toAttr: AttrValue = dataTypeToAttrValue(dt)
  }

  def buildShape(s: Shape): TensorShapeProto = s.toProto

  private def buildType(sqlType: NumericType): AttrValue = {
    AttrValue.newBuilder().setType(getDType(sqlType)).build()
  }

  def buildGraph(nodes: Seq[Node]): GraphDef = {
    logTrace("buildGraph: freezing nodes")
    // This only uses the parents that are internally reachable
    // TODO(tjh) this freezing business is not going to work.
    // We need to store the creation order (and reconstruct it for subnodes using a creation tree)
    nodes.foreach(_.freeze())
    // After that, we also need to freeze some constants and other elements around.
    logTrace("buildGraph: Freezing everything")
    nodes.foreach(_.freeze(everything=true))
    logTrace(s"buildGraph for nodes: ${nodes.map(_.name)}")
    var treated: Map[String, Node] = Map.empty
    nodes.foreach { n =>
      logTrace(s"call: n=${n.name}, treated=${treated.keySet}")
      treated = getClosure(n, treated)
    }
    val b = GraphDef.newBuilder()
    treated.values.flatMap(_.nodes).foreach(b.addNode)
    b.build()
  }

  def buildGraph(node: Node, nodes: Node*): GraphDef = {
    buildGraph(Seq(node) ++ nodes)
  }

  private def getClosure(node: Node, treated: Map[String, Node]): Map[String, Node] = {
    logTrace(s"closure: n=${node.name}, parents=${node.parents.map(_.name)}," +
      s" treated=${treated.keySet}")
    val explored = node.parents
      .filterNot(n => treated.contains(n.name))
      .flatMap(getClosure(_, treated + (node.name -> node)))
      .toMap

    uniqueByName(node +: (explored.values.toSeq ++ treated.values.toSeq))
  }

  private def uniqueByName(nodes: Seq[Node]): Map[String, Node] = {
    nodes.groupBy(_.name).mapValues(_.head)
  }

  def build_constant(dt: DenseTensor): Node = {
    val a = AttrValue.newBuilder().setTensor(DenseTensor.toTensorProto(dt))
    build("Const", isOp = false,
      shape = dt.shape, dtype = dt.dtype,
      extraAttrs = Map("value" -> a.build()))
  }

  private[tensorframes] def placeholder(dtype: NumericType, shape: Shape): Node = {
    build("Placeholder", shape=shape, dtype=dtype, isOp = false,
      extraAttrs = Map("shape" -> shape.toAttr))
  }

  private[dsl] def extractPlaceholder(
      df: DataFrame,
      colName: String,
      tfName: String,
      block: Boolean): Operation = {
    val schema = df.schema.find(_.name == colName).getOrElse {
      throw new Exception(s"Cannot find column $colName, available columns" +
        s" are ${df.schema.fieldNames.mkString(",")}")
    }
    // Get the shape and dtype
    val stf = ColumnInformation(schema).stf.getOrElse {
      throw new Exception(s"The datatype of column '$colName' could not be understood by" +
        s"tensorframes: $schema")
    }
    val shape = if (block) { stf.shape } else { stf.shape.tail }
    DslImpl.placeholder(stf.dataType, shape).named(tfName)
  }

  private def commonShape(shapes: Seq[Shape]): Shape = {
    require(shapes.nonEmpty)
    require(shapes.forall(_ == shapes.head), s"$shapes")
    shapes.head
  }

  private val U = Unknown.toLong

  // Implements the broadcasting rules
  private[dsl] def broadcastShape(shapes: Seq[Shape]): Shape = {
    require(shapes.length == 2, shapes)
    val Seq(s1, s2) = shapes
    if (s1.numDims < s2.numDims) {
      broadcastShape(Seq(s2, s1))
    } else {
      // The head is going to be the same
      val t = s1.dims.take(s1.numDims - s2.numDims)
      // No need to work reverse, because we have isolated the head.
      val h = (s1.dims zip s2.dims).map {
        case (d1, d2) if d1 == U || d1 == 1L => d2
        case (d1, d2) if d2 == U || d2 == 1L => d1
        case (d1, d2) if d1 == d2 => d1
        case _ => throw new Exception(s"Incompatible shapes: $s1 $s2")
      }
      Shape((t ++ h).toArray)
    }
  }

  private def commonType(dtypes: Seq[NumericType]): NumericType = {
    require(dtypes.nonEmpty)
    require(dtypes.forall(_ == dtypes.head), s"All these types should be the same: $dtypes")
    dtypes.head
  }

  def build(
      opName: String,
      @Nullable name: String = null,
      parents: Seq[Node] = Seq.empty,
      extraParents: String => Seq[Node] = _ => Nil,
      @Nullable isOp: Boolean = true,
      @Nullable dtype: NumericType = null,
      @Nullable shape: Shape = null,
      dtypeInfer: Seq[NumericType] => NumericType = commonType,
      shapeInfer: Seq[Shape] => Shape = commonShape,
      extraAttrs: Map[String, AttrValue] = Map.empty): Node = {
    val dt = Option(dtype).getOrElse(dtypeInfer(parents.map(_.scalarType)))
    val sh = Option(shape).getOrElse(shapeInfer(parents.map(_.shape)))
    Node(Option(name), opName, dt, sh, parents, extraParents, isOp, extraAttrs)
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

  def build_reducer(
      opName: String,
      parent: Node,
      reduction_indices: Seq[Int] = null,
      name: String = null): Node = {
    val idxs = constant(reduction_indices) named (parent.name + "/reduction_indices")
    val attr = AttrValue.newBuilder().setB(false).build()
    build(opName, name, Seq(parent, idxs),
      dtype = parent.scalarType,
      shape = reduce_shape(parent.shape, Option(reduction_indices).getOrElse(Nil)),
      extraAttrs = Map(
        "Tidx" -> AttrValue.newBuilder().setType(DataType.DT_INT32).build(),
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