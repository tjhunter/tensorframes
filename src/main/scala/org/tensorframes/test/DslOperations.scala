package org.tensorframes.test

import scala.collection.JavaConverters._

import org.apache.spark.sql.{RelationalGroupedDataset, DataFrame, Row}
import org.tensorflow.framework.GraphDef
import org.tensorframes.{OperationsInterface, ShapeDescription}
import org.tensorframes.impl.{GraphNodeSummary, TensorFlowOps}
import org.tensorframes.test.dsl.Node

/**
 * Convenience API for DSL-based operations.
 *
 * This is more convenient than using the standard API when working with the DSL.
 */
trait DslOperations extends OperationsInterface {

  import DslOperations._

  private def make[A](
      ns: Seq[Node],
      df: DataFrame,
      f: (DataFrame, GraphDef, ShapeDescription) => A): A = {
    val g = dsl.buildGraph(ns: _*)
    val info = extraInfo(ns, g)
    f(df, g, info)
  }

  def mapBlocks(df: DataFrame, node1: dsl.Node, nodes: dsl.Node*): DataFrame = {
    val ns = node1 +: nodes
    make(ns, df, mapBlocks)
  }

  def mapRows(df: DataFrame, node1: dsl.Node, nodes: dsl.Node*): DataFrame = {
    val ns = node1 +: nodes
    make(ns, df, mapRows)
  }

  def reduceRows(df: DataFrame, node1: dsl.Node, nodes: dsl.Node*): Row = {
    val ns = node1 +: nodes
    make(ns, df, reduceRows)
  }

  def reduceBlocks(df: DataFrame, node1: dsl.Node, nodes: dsl.Node*): Row = {
    val ns = node1 +: nodes
    make(ns, df, reduceBlocks)
  }

  def aggregate(gdf: RelationalGroupedDataset, node1: dsl.Node, nodes: dsl.Node*): DataFrame = {
    val ns = node1 +: nodes
    val g = dsl.buildGraph(ns: _*)
    val info = extraInfo(ns, g)
    aggregate(gdf, g, info)
  }
}

object DslOperations {

  private def extraInfo(fetches: Seq[Node], g: GraphDef): ShapeDescription = {
    val inputs = graphInputs(g).map(x => x -> x).toMap
    ShapeDescription(
      fetches.map(n => n.name -> n.shape).toMap,
      fetches.map(_.name),
      inputs)
  }

  def analyzeGraph(nodes: Node*): (GraphDef, Seq[GraphNodeSummary]) = {
    val g = dsl.buildGraph(nodes: _*)
    g -> TensorFlowOps.analyzeGraphTF(g, extraInfo(nodes, g))
  }

  def graphInputs(g: GraphDef): Seq[String] = {
    g.getNodeList.asScala.filter(_.getInputCount == 0)
      .map(_.getName)
  }

}