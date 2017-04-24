package org.tensorframes

import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.scalatest.FunSuite
import org.tensorframes.impl.{ScalarDoubleType, ScalarIntType}


class ExtraOperationsSuite
    extends FunSuite with TensorFramesTestSparkContext with Logging {
  lazy val sql = sqlContext
  import ExtraOperations._
  import sql.implicits._
  import Shape.Unknown

  test("simple test for doubles") {
    val df = Seq(Tuple1(0.0)).toDF("a")
    val di = ExtraOperations.explainDetailed(df)
    val Seq(c1) = di.cols
    val Some(s) = c1.stf
    assert(s.dataType === ScalarDoubleType)
    assert(s.shape === Shape(Unknown))
    logDebug(df.toString() + "->" + di.toString)
  }

  test("simple test for integers") {
    val df = Seq(Tuple1(0)).toDF("a")
    val di = explainDetailed(df)
    val Seq(c1) = di.cols
    val Some(s) = c1.stf
    assert(s.dataType === ScalarIntType)
    assert(s.shape === Shape(Unknown))
    logDebug(df.toString() + "->" + di.toString)
  }

  test("test for arrays") {
    val df = Seq((0.0, Seq(1.0), Seq(Seq(1.0)))).toDF("a", "b", "c")
    val di = explainDetailed(df)
    logDebug(df.toString() + "->" + di.toString)
    val Seq(c1, c2, c3) = di.cols
    val Some(s1) = c1.stf
    assert(s1.dataType === ScalarDoubleType)
    assert(s1.shape === Shape(Unknown))
    val Some(s2) = c2.stf
    assert(s2.dataType === ScalarDoubleType)
    assert(s2.shape === Shape(Unknown, Unknown))
    val Some(s3) = c3.stf
    assert(s3.dataType === ScalarDoubleType)
    assert(s3.shape === Shape(Unknown, Unknown, Unknown))
  }

  test("simple analysis") {
    val df = Seq(Tuple1(0.0)).toDF("a")
    val df2 = analyze(df)
    val di = explainDetailed(df2)
    logDebug(df.toString() + "->" + di.toString)
    val Seq(c1) = di.cols
    val Some(s) = c1.stf
    assert(s.dataType === ScalarDoubleType)
    assert(s.shape === Shape(1)) // There is only one partition
  }

  test("simple analysis with multiple partitions of different sizes") {
    val df = Seq.fill(10)(0.0).map(Tuple1.apply).toDF("a").repartition(3)
    val df2 = analyze(df)
    val di = explainDetailed(df2)
    logDebug(df.toString() + "->" + di.toString)
    val Seq(c1) = di.cols
    val Some(s) = c1.stf
    assert(s.dataType === ScalarDoubleType)
    assert(s.shape === Shape(Unknown)) // There is only one partition
  }

  test("simple analysis with variable sizes") {
    val df = Seq(
      (0.0, Seq(0.0)),
      (1.0, Seq(1.0, 1.0))).toDF("a", "b")
    val df2 = analyze(df)
    val di = explainDetailed(df2)
    logDebug(df.toString() + "->" + di.toString)
    val Seq(c1, c2) = di.cols
    val Some(s2) = c2.stf
    assert(s2.dataType === ScalarDoubleType)
    assert(s2.shape === Shape(2, Unknown)) // There is only one partition
  }

  test("2nd order analysis") {
    val df = Seq(
      (0.0, Seq(0.0, 0.0)),
      (1.0, Seq(1.0, 1.0)),
      (2.0, Seq(2.0, 2.0))).toDF("a", "b")
    val df2 = analyze(df)
    val di = explainDetailed(df2)
    logDebug(df.toString() + "->" + di.toString)
    val Seq(c1, c2) = di.cols
    val Some(s2) = c2.stf
    assert(s2.dataType === ScalarDoubleType)
    assert(s2.shape === Shape(3, 2)) // There is only one partition
  }
}
