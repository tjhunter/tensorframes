package org.tensorframes.impl

import org.apache.spark.sql.types.NumericType
import org.bytedeco.javacpp.{BytePointer, tensorflow => jtf}
import org.tensorflow.framework.GraphDef
import org.tensorframes.test.ProtoConversions
import org.tensorframes.{Logging, Shape, ShapeDescription}

import scala.collection.JavaConverters._

/**
 * Some low-level tensorflow operations.
 */
object TensorFlowOps extends Logging {

  private[this] val lock = new Object

  lazy val _init = lock.synchronized {
    logDebug("Starting TensorFlowOps...")
    jtf.InitMain("test", Array.empty[Int], null)
    logDebug("Starting TensorFlowOps... Done")
    true
  }

  def initTensorFlow(): Unit = {
    _init
  }

  def graphSerial(g: jtf.GraphDef): Array[Byte] = {
    val n = g.ByteSizeLong()
    assert(n < Int.MaxValue, s"Cannot serialize graphs of size more than ${Int.MaxValue} " +
      s"(trying to serialize a graph of size $n bytes")
    val arr = Array.fill[Byte](g.ByteSizeLong().toInt)(0)
    g.SerializeWithCachedSizesToArray(arr)
    arr
  }

  def graphSerial(g: GraphDef): Array[Byte] = {
    g.toByteString.toByteArray
  }

  def readGraphSerial(arr: Array[Byte]): GraphDef = {
    GraphDef.parseFrom(arr)
  }

  def readGraph(arr: Array[Byte]): jtf.GraphDef = {
    val res = new jtf.GraphDef()
    val p = new BytePointer(arr.length)
    p.put(arr, 0, arr.length)
    jtf.ParseProtoUnlimited(res, p)
    res
  }


  def withSession[T](f: jtf.Session => T): T = {
    initTensorFlow()
    val options = new jtf.SessionOptions()
    val session = new jtf.Session(options)
    try {
      f(session)
    } finally {
      session.Close()
    }
  }

  def jtfShape(s: jtf.TensorShape): Shape = {
    val dims = (0 until s.dims()).map(s.dim_size).toArray
    Shape(dims)
  }

  def shape(sh: Shape): jtf.TensorShape = {
    val s = new jtf.TensorShape()
    sh.dims.foreach { dim =>
      s.AddDim(dim)
    }
    s
  }


  /**
   * Performs some analysis over the TF graph, by loading it into the TF runtime and extracting
   * the shapes of the various components in it.
   */
  def analyzeGraph(
      graphDef: GraphDef,
      shapeHints: ShapeDescription = ShapeDescription.empty): Seq[GraphNodeSummary] = {
    initTensorFlow()
//    logTrace(s"analyzeGraph: shapeHints=$shapeHints")
//    logTrace(s"analyzeGraph: graph=$graphDef")

    val nodes = graphDef.getNodeList.asScala
    val inputs: Set[String] = nodes
      .filter(n => n.getInputCount == 0 && n.getOp == "Placeholder")
      .map(_.getName).toSet
    // We identify a node with its output tensor.
    val outputs = shapeHints.requestedFetches.map(_.stripSuffix(":0")).toSet
    logDebug(s"Outputs: ${outputs}")

    withSession { session =>
      val g = readGraph(graphSerial(graphDef))
      val s1 = session.Extend(g)
      assert(s1.ok(), s1.error_message().getString)
      val options = new jtf.GraphConstructorOptions()
      val registry = jtf.OpRegistry.Global()
      val graph = new jtf.Graph(registry)
      val s2 = jtf.ConvertGraphDefToGraph(options, g, graph)
      val nodes = {
        val x = graph.nodes()
        var res: List[jtf.Node] = Nil
        val it = x.begin()
        while (it.notEquals(x.end())) {
          res ::= it.access()
          it.increment()
        }
        res
      }
      logDebug(s"Extracted ${nodes.size} nodes")
      // TODO: move this within the iterator, the nodes it attempts to access may have been deallocated at that point.
      nodes.filter(_.id() >= 2).map { node =>
        val id = node.id()
        val name = "?"
//        val name = node.name() // crash
        val op = "" //node.op_def().name().getString
//        val outputs = {
//          val l = node.output_types().size().toInt
//          (0 until 0).map { idx =>
//            node.output_type(idx)
//          }
//        }
        logDebug(s"Node: id=$id name=$name op=$op debug=${node.DebugString().getString}")
      }
      assert(s2.ok(), s2.error_message().getString)
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

  def stringVector(strings: Seq[String]): jtf.StringVector = {
    val o = new jtf.StringVector(strings.length)
    strings.indices.foreach { idx =>
      o.put(idx, strings(idx))
    }
    o
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
