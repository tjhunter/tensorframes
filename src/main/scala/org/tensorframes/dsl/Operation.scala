package org.tensorframes.dsl

import org.apache.spark.sql.types.NumericType
import org.tensorflow.framework.{AttrValue, GraphDef, NodeDef}
import org.tensorframes.test.DslOperations
import org.tensorframes.{Logging, Shape, ShapeDescription, dsl => tf}

import scala.collection.JavaConverters._

/**
 * A node in the TensorFlow graph.
 *
 * There is currently no difference between nodes and the default tensor output.
 */
trait Operation {
  /**
   * The path of the operation. It follows TensorFlow's path conventions.
   *
   * @return
   */
  def name: String

  /**
   * The name of the operation.
   *
   * @return
   */
  def opName: String

  /**
   * The dimensions of the default tensor output of this operation.
   *
   * @return
   */
  def dims: Seq[Int]

  /**
   * Use this method if you want to give a name to a node. This method can only be called once
   * on a node.
   *
   * Note that the name will automatically include the current path, if the path is changed with
   * [[Paths]].
   *
   * @param newName the name of the node.
   * @return a named operation.
   */
  def named(newName: String): Operation

  /**
   * Point-wise addition (with broadcasting).
   */
  def +(other: Operation): Operation = tf.add(this, other)

  /**
   * Point-wise division (with broadcasting).
   */
  def /(other: Operation): Operation = tf.div(this, other)
}

/**
 * Implementation of an operation.
 */
private[tensorframes] case class Node(
    requestedName: Option[String], // The name given by the user
    creationPath: List[String], // The context path when this node was created
    opName: String,
    scalarType: NumericType,
    shape: Shape,
    parents: Seq[Node],
    internalParents: String => Seq[Node],
    isOp: Boolean,
    extraAttr: Map[String, AttrValue]) extends Operation with Logging {

  import ProtoConversions._
  import DslImpl._

  private var _path: String = null
  private var _createdParents: Seq[Node] = null
  private val creationStack = (new Exception).getStackTraceString
  private var frozenStack: String = null

  logTrace(s"Created node $this")

  private def frozen = { _path != null }

  def freeze(everything: Boolean = false): Unit = {
    logTrace(s"Calling freeze on $this with ${_path}")
    // May increment counter as a side effect here.
    if (! frozen) {
      val p = Paths.path(creationPath, requestedName, opName)
      _path = p
      logTrace(s"freeze: Path $p for $this")
      frozenStack = (new Exception).getStackTraceString
      val diff = creationPath.filterNot(_.isEmpty).mkString("/")
      val suffix = p.drop(diff.length)
      val ns = internalParents(suffix)
      ns.foreach(_.freeze())
      _createdParents = ns
    }
    if (everything) {
      allParents.foreach(_.freeze(everything))
    }
    assert(frozen)
  }

  private def allParents = {
    require(frozen)
    parents ++ _createdParents
  }

  def name: String = {
    require(frozen, s"The node $this is not frozen. Creation stack:\n$creationStack")
    _path
  }

  override def dims: Seq[Int] = shape.dims.map(_.toInt)

  // The node and all its internal nodes
  def nodes: Seq[NodeDef] = {
    freeze()
    val b = NodeDef.newBuilder()
    b.setName(name)
    b.setOp(opName)
    allParents.foreach(p => b.addInput(p.name))
    if (isOp) {
      b.putAllAttr(Map("T" -> getDType(scalarType).toAttr).asJava)
    } else {
      b.putAllAttr(Map(
        "dtype" -> getDType(scalarType).toAttr).asJava)
    }
    b.putAllAttr(extraAttr.asJava)
    Seq(b.build()) ++ _createdParents.flatMap(_.nodes)
  }

  override def named(newName: String): Node = {
    val c = copy(requestedName = Some(newName))
    c.freeze()
    c
  }

  override def toString(): String = {
    val f = if (frozen) {"frz"} else {"liv"}
    s"Node($f${math.abs(this.##)}, $requestedName, $creationPath, $opName, $scalarType, $shape)"
  }

}

private[dsl] object Node {

  def apply(
      requestedName: Option[String],
      opName: String,
      scalarType: NumericType,
      shape: Shape,
      parents: Seq[Node],
      internalParents: String => Seq[Node],
      isOp: Boolean,
      extraAttr: Map[String, AttrValue]) = {
    val p = Paths.creationPath()
    new Node(requestedName, p, opName, scalarType, shape, parents, internalParents, isOp, extraAttr)
  }

  /**
   * Builds shape hints from the shape description.
   */
  def hints(ns: Seq[Node], g: GraphDef): ShapeDescription = {
    val m = ns.map { n =>
      n.name -> n.shape
    } .toMap
    val f = ns.map(_.name)
    // Use the inputs as names.
    val inputs = DslOperations.graphInputs(g).map(x => x -> x).toMap

    ShapeDescription(m, f, inputs)
  }

}
