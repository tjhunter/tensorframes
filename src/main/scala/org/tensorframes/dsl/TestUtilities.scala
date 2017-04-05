package org.tensorframes.dsl

import java.nio.file.{Files, Paths => JPaths}

import org.tensorflow.framework.GraphDef

import org.tensorframes.ShapeDescription
import org.tensorframes.impl.{GraphNodeSummary, TensorFlowOps}

/**
 * Some utilities for running tests.
 */
// TODO(tjh) check that these methods are not implemented somewhere else.
private[tensorframes] object TestUtilities {

  def buildGraph(node: Operation, nodes: Operation*): GraphDef = {
    DslImpl.buildGraph(Seq(node) ++ nodes)
  }

  def loadGraph(file: String): GraphDef = {
    val byteArray = Files.readAllBytes(JPaths.get(file))
    GraphDef.newBuilder().mergeFrom(byteArray).build()
  }

  private def extraInfo(fetches: Seq[Node]): ShapeDescription = {
    val m2 = fetches.map(n => n.name -> n.name).toMap
    ShapeDescription(
      fetches.map(n => n.name -> n.shape).toMap,
      fetches.map(_.name),
      m2)
  }


  def analyzeGraph(nodes: Operation*): (GraphDef, Seq[GraphNodeSummary]) = {
    val g = buildGraph(nodes.head, nodes.tail: _*)
    g -> TensorFlowOps.analyzeGraph(g, extraInfo(nodes))
  }

}
