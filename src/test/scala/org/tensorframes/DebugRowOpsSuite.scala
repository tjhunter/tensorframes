package org.tensorframes

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.scalatest.FunSuite
import org.tensorframes.impl.{DebugRowOpsImpl, ScalarDoubleType}
import org.tensorframes.dsl._

class DebugRowOpsSuite
  extends FunSuite with TensorFramesTestSparkContext with GraphScoping with Logging {
  lazy val sql = sqlContext
  import ColumnInformation.structField
  import Shape.Unknown

  testGraph("Simple identity") {
    val rows = Array(Row(1.0))
    val input = StructType(Array(structField("x", ScalarDoubleType, Shape(Unknown))))
    val p2 = placeholder[Double](1) named "x"
    val out = identity(p2) named "y"
    val outputSchema = StructType(Array(structField("y", ScalarDoubleType, Shape(Unknown))))
    val (g, _) = TestUtilities.analyzeGraph(out)
    logDebug(g.toString)
    val res = DebugRowOpsImpl.performMap(rows, input, Array("x" -> 0), g, outputSchema)
    assert(res === Array(Row(1.0, 1.0)))
  }

  testGraph("Simple add") {
    val rows = Array(Row(1.0))
    val input = StructType(Array(structField("x", ScalarDoubleType, Shape(Unknown))))
    val p2 = placeholder[Double](1) named "x"
    val out = p2 + p2 named "y"
    val outputSchema = StructType(Array(structField("y", ScalarDoubleType, Shape(Unknown))))
    val (g, _) = TestUtilities.analyzeGraph(out)
    logDebug(g.toString)
    val res = DebugRowOpsImpl.performMap(rows, input, Array("x" -> 0), g, outputSchema)
    assert(res === Array(Row(2.0, 1.0)))
  }

}
