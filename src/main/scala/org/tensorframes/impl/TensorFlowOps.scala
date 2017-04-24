package org.tensorframes.impl

import java.io.File
import java.nio.file.{Files, Paths}

import org.tensorflow.framework.GraphDef
import org.{tensorflow => tf}
import org.apache.spark.sql.types.NumericType
import org.tensorflow.{Graph, Session}
import org.tensorframes.test.ProtoConversions
import org.tensorframes.{Logging, Shape, ShapeDescription}

import scala.collection.JavaConverters._

/**
  * Contains a TensorFlow graph that has been serialized using protocol buffers.
  *
  * In order to limit the amount of memory being used by this class, it has the ability to dump its content onto
  * a disk file, and serve the data from this disk file.
  */
case class SerializedGraph private (
  private var _content: Option[Array[Byte]],
  private var file: Option[String]) extends Serializable with Logging {

  def content: Array[Byte] = file match {
    case Some(name) =>
      val p = Paths.get(name)
      Files.readAllBytes(p)
    case None =>
      _content.getOrElse {
        throw new Exception(s"Missing content for serialized graph $this")
      }
  }

  /**
    * Moves the graph description to a file, and drops the in-memory representation once it is safe to do so.
    */
  def evictContent(): Unit = this.synchronized {
    if (file.isDefined) {
      return // Nothing to do here
    }
    val bytes = _content.getOrElse {
      throw new Exception(s"Missing content for serialized graph $this")
    }
    val tempFile = File.createTempFile("tensorframes-graphs-", "-proto-bin")
    tempFile.deleteOnExit()
    SerializedGraph.logInfo(s"Evicting graph to temporary file $tempFile...")
    Files.write(tempFile.toPath, bytes)
    file = Some(tempFile.toString)
    _content = None
    SerializedGraph.logInfo(s"Done evicting graph graph: $this")
  }
}

object SerializedGraph extends Logging {
  // Stored in memory by default, so that the broadcast mechanism can send it around.
  def create(content: Array[Byte]): SerializedGraph = {
    require(content != null)
    new SerializedGraph(Some(content), None)
  }
}

/**
 * Some low-level tensorflow operations.
 */
object TensorFlowOps extends Logging {

  def graphSerial(g: GraphDef): SerializedGraph = {
    SerializedGraph.create(g.toByteString.toByteArray)
  }

  def readGraphSerial(arr: SerializedGraph): GraphDef = {
    GraphDef.parseFrom(arr.content)
  }

  def withSession[T](g: SerializedGraph)(f: tf.Session => T): T = {
    withGraph(g) { graph =>
      val session = new Session(graph)
      try {
        f(session)
      } finally {
        session.close()
      }
    }
  }

  def withGraph[T](g: SerializedGraph)(f: tf.Graph => T): T = {
    val graph2 = new Graph()
    graph2.importGraphDef(g.content)
    try {
      f(graph2)
    } finally {
      graph2.close()
    }
  }

  /**
    * Performs some analysis over the TF graph, by loading it into the TF runtime and extracting
    * the shapes of the various components in it.
    */
  def analyzeGraphTF(
      graphDef: GraphDef,
      shapeHints: ShapeDescription = ShapeDescription.empty): Seq[GraphNodeSummary] = {

    val nodes = graphDef.getNodeList.asScala
    val inputs: Set[String] = nodes
      .filter(n => n.getInputCount == 0 && n.getOp == "Placeholder")
      .map(_.getName).toSet
    // We identify a node with its output tensor.
    val outputs = shapeHints.requestedFetches.map(_.stripSuffix(":0")).toSet
    logDebug(s"Outputs: ${outputs}")

    // Test that the graph can be imported
    val sg = graphSerial(graphDef)
    withGraph(sg){ g =>
      logDebug(s"analyzeGraphTF: the graph has size ${sg.content.length.toLong/1000000} MB and ${nodes.size} nodes")
      nodes.flatMap { n =>
        val name = n.getName
        val op = g.operation(name)
        val isInput = inputs.contains(name)
        val isOutput = outputs.contains(name)
        if (isInput || isOutput) {
          // Shape: this is tricky since dynamic shapes get completely pruned out of the compute graph (as of TF 1.0)
          // Hints are still required to recover the shape.
          // The syntax is not very clear here: it could either refer to the first tensor, or to the operator.
          val hintedShape = shapeHints.out.get(name).orElse(shapeHints.out.get(name+":0"))
          // NOTE: this only considers the first output of an operation
          // The user cannot currently refer to other tensors.
          val res = getSummaryDefault(op).headOption.map { case (scalarType, shape) =>
            // The shape provided in the hints overrides the shape inferred from the graph, since that one may
            // be missing the dynamic shapes.
            val s = hintedShape.getOrElse(shape)
            GraphNodeSummary(isInput, isInput, isOutput, scalarType, s, name)
          }
          res
        } else {
          Nil
        }
      }
    }
  }

  private def getSummaryDefault(op: tf.Operation): Seq[(ScalarType, Shape)] = {
    (0 until op.numOutputs()).map { idx =>
      val n = op.output(idx)
      val dt = SupportedOperations.opsFor(n.dataType()).scalarType
      val shape = Shape.from(n.shape())
      dt -> shape
    }
  }
}

/**
 * All the information requested by TensorFrames to run on a graph node.
 *
 * @param isPlaceholder if the variable is a placeholder
 * @param isInput if the node is an input (no inner dependencies)
 * @param isOutput if it is an outpu (no node depends on it)
 * @param scalarType the scalar type of the final tensor associated to this node
 * @param shape the shape of the final tensor associated to this node
 * @param name the name of the node in the graph
 */
case class GraphNodeSummary(
    isPlaceholder: Boolean,
    isInput: Boolean,
    isOutput: Boolean,
    scalarType: ScalarType,
    shape: Shape,
    name: String) extends Serializable
