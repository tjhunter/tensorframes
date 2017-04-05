package org.tensorframes.dsl

import scala.languageFeature.implicitConversions

import scala.collection.JavaConverters._

import org.apache.spark.sql.{RelationalGroupedDataset, Row, DataFrame}
import org.tensorflow.framework.GraphDef
import org.tensorframes.{ExperimentalOperations, OperationsInterface, ShapeDescription, dsl}

/**
 * Implicit transforms to help with the construction of TensorFrames manipulations.
 */
trait DFImplicits {
  protected def ops: OperationsInterface with ExperimentalOperations

  /**
   * This implicit augments Spark's DataFrame with a number of tensorflow-related methods.
   *
   * These methods are the preferred way to manipulate DataFrames, see the documentation for
   * examples.
   *
   * @param df the underlying dataframe.
   */
  implicit class RichDataFrame(df: DataFrame) {

    def mapRows(graph: GraphDef, shapeHints: ShapeDescription): DataFrame = {
      ops.mapRows(df, graph, shapeHints)
    }

    def mapRows(o0: Operation, os: Operation*): DataFrame = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(seq)
      mapRows(g, Node.hints(seq, g))
    }

    def mapBlocks(graph: GraphDef, shapeHints: ShapeDescription): DataFrame = {
      ops.mapBlocks(df, graph, shapeHints)
    }

    def mapBlocks(o0: Operation, os: Operation*): DataFrame = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(Seq(o0) ++ os)
      mapBlocks(g, Node.hints(seq, g))
    }

    def mapBlocksTrimmed(graph: GraphDef, shapeHints: ShapeDescription): DataFrame = {
      ops.mapBlocksTrimmed(df, graph, shapeHints)
    }

    def mapBlocksTrimmed(o0: Operation, os: Operation*): DataFrame = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(Seq(o0) ++ os)
      mapBlocksTrimmed(g, Node.hints(seq, g))
    }

    def reduceRows(graph: GraphDef, shapeHints: ShapeDescription): Row = {
      ops.reduceRows(df, graph, shapeHints)
    }

    def reduceRows(o0: Operation, os: Operation*): Row = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(Seq(o0) ++ os)
      reduceRows(g, Node.hints(seq, g))
    }

    def reduceBlocks(graph: GraphDef, shapeHints: ShapeDescription): Row = {
      ops.reduceBlocks(df, graph, shapeHints)
    }

    def reduceBlocks(o0: Operation, os: Operation*): Row = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(Seq(o0) ++ os)
      reduceBlocks(g, Node.hints(seq, g))
    }

    def explainTensors: String = ops.explain(df)

    def analyze(): DataFrame = {
      ops.analyze(df)
    }

    // TODO: do we need this? it can be named.
    def row(columnName: String, tfName: String): Operation = {
      dsl.row(df, columnName, tfName)
    }

    def row(columnName: String): Operation = {
      dsl.row(df, columnName, columnName)
    }

    // TODO: do we need this? it can be named.
    def block(columnName: String, tfName: String): Operation = {
      dsl.block(df, columnName, tfName)
    }

    def block(columnName: String): Operation = {
      dsl.block(df, columnName, columnName)
    }
  }

  /**
   * Extra operations for Spark's RelationalGroupedDataset.
   *
   * This is useful for aggregation.
   */
  implicit class RichRelationalGroupedDataset(dg: RelationalGroupedDataset) {
    def aggregate(graphDef: GraphDef, shapeDescription: ShapeDescription): DataFrame = {
      ops.aggregate(dg, graphDef, shapeDescription)
    }
    def aggregate(o0: Operation, os: Operation*): DataFrame = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(Seq(o0) ++ os)
      aggregate(g, Node.hints(seq, g))
    }
  }

  /**
   * Automatically converts constants to TensorFlow nodes.
   */
  implicit def canConvertToConstant[T : ConvertibleToDenseTensor](x: T): Operation = {
    dsl.constant(x)
  }

}

/**
 * You should import this object if you want to access all the TensorFrames DSL.
 */
object Implicits extends DFImplicits with DefaultConversions {
  protected override def ops = Ops
}
