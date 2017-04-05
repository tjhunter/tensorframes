package org.tensorframes

import org.scalatest.FunSuite
import org.tensorframes.impl.TensorFlowOps


class TFInitializationSuite extends FunSuite with Logging {

  import dsl._
  import dsl.TestUtilities._

  test("Simple loading of a placeholder") {
    val p1 = placeholder[Double](1) named "p1"
    val p2 = placeholder[Double](1) named "p2"
    val a = p1 + p2 named "a"
    val g = buildGraph(a)
    logDebug(g.toString)
    TensorFlowOps.analyzeGraphTF(g, ShapeDescription(Map("a"->Shape(1)), Seq("a"), Map("a"->"a")))
  }

  // Requires support for floats
  // TODO(tjh) activate when floats are supported
  ignore("Loading an existing graph") {
    val g2 = loadGraph("/home/tensorframes/src/test/resources/graph2.pb")
    logDebug(g2.toString)
    TensorFlowOps.analyzeGraphTF(g2)
  }

  test("Graph with identity") {
    val p2 = placeholder[Double](1) named "x"
    val out = identity(p2) named "y"
    val g = buildGraph(out)
    TensorFlowOps.analyzeGraphTF(g, ShapeDescription(Map("y"->Shape(1)), Seq("y"), Map("a"->"a")))
  }
}