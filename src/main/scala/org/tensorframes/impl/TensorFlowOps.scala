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
    val graph2 = new Graph()
    graph2.importGraphDef(g.content)
    val session = new Session(graph2)
    try {
      f(session)
    } finally {
      session.close()
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
    {
//      val g = new Graph()
      val ser = graphSerial(graphDef).content
      logInfo(s"analyzeGraphTF: the graph has size ${ser.length.toLong/1000000} MB")
//      g.importGraphDef(ser)
//      g.close()
    }

    nodes.flatMap { n =>
      val name = n.getName
      logTrace(s"Node $name")
      val isInput = inputs.contains(name)
      val isOutput = outputs.contains(name)
      if (isInput || isOutput) {
        // The shape stored in the graph seems buggy sometimes (when there are some unknowns)
        // Trust the one from the shape hints.
        val shapeOpt = shapeHints.out.get(name).orElse {
          // The name may include the default output slot
          // TODO(tjh) add a test for that
          shapeHints.out.get(name + ":0")
        } .orElse {
          if (n.getAttr.containsKey("shape")) {
            Some(Shape.from(n.getAttr.get("shape").getShape))
          } else {
            None
          }
        }
        logTrace(s"shape = $shapeOpt")
        val shape = shapeOpt.getOrElse {
          throw new Exception(s"Could not get the shape of node $name from the graph definition or from the shape hints")
        }
        val scalarType = SupportedOperations.opsFor(ProtoConversions.getDType(n)).sqlType
        Some(GraphNodeSummary(isInput, isInput, isOutput, scalarType, shape, name))
      } else { None }
    }
  }
}

/**
 * All the informations requested by TensorFrames to run on a graph node.
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
    scalarType: NumericType,
    shape: Shape,
    name: String) extends Serializable
