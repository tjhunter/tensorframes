package org.tensorframes

import org.apache.spark.sql.Row
import org.tensorframes.test.dsl._


// These tests do not require any operation to run.
trait BasicIdentityTests[T] { self: CommonOperationsSuite[T] =>
  import self._

  test(s"Identity $dtname") {
    Seq(1.0, 20.0).u
    val df = make1(Seq(1.0, 20.0).u, "in")
    val p1 = placeholder(dtype, Shape(Unknown)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapBlocks(df, out).select("in", "out")
    assert(df2.collect() === Seq(Row(1.0, 1.0), Row(20.0, 20.0)).u)
  }

  test(s"Identity - 1 dim $dtname") {
    val df = make1(Seq(Seq(1.0), Seq(20.0)).u, "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(dtype, Shape(Unknown, 1)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapBlocks(adf, out).select("in", "out")
    assert(df2.collect() === Seq(Row(Seq(1.0), Seq(1.0)), Row(Seq(20.0), Seq(20.0))).u)
  }

  test(s"Identity - 1 known dim $dtname") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(20.0)).u, "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(dtype, Shape(1)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(adf, out).select("in", "out")
    compareRows(df2.collect(), Seq(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(20.0), Seq(20.0))).u)
  }

  test(s"Identity - 1 unknown dim $dtname") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(20.0)).u, "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(dtype, Shape(Shape.Unknown)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(adf, out).select("in", "out")
    compareRows(df2.collect(), Seq(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(20.0), Seq(20.0))).u)
  }

  test(s"Identity - 1 dim and variable sizes $dtname") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(20.0, 21.0)).u, "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(dtype, Shape(Shape.Unknown)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(adf, out).select("in", "out")
    compareRows(df2.collect(), Seq(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(20.0, 21.0), Seq(20.0, 21.0))).u)
  }

}

// These tests require a working add operation
trait BasicMonoidTests[T] { self: CommonOperationsSuite[T] =>
  import self._

  test(s"Simple add $dtname") {
    val df = sql.createDataFrame(Seq(10.0->11.0, 20.0->22.0).u).toDF("a", "b")
    val a = placeholder(dtype, Shape(Unknown)) named "a"
    val b = placeholder(dtype, Shape(Unknown)) named "b"
    val out = a + b named "out"
    val df2 = ops.mapBlocks(df, out).select("a", "b","out")
    compareRows(df2.collect(), Seq(Row(10.0, 11.0, 21.0), Row(20.0, 22.0, 42.0)).u)
  }

  test(s"Simple add - 1 dim $dtname") {
    val a = placeholder(dtype, Shape(Unknown, 1)) named "a"
    val b = placeholder(dtype, Shape(Unknown, 1)) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(10.0)->Seq(11.0),
      Seq(20.0)->Seq(22.0)).u).toDF("a", "b")
    val adf = ops.analyze(df)
    val df2 = ops.mapBlocks(adf, out).select("a", "b","out")
    compareRows(df2.collect(), Seq(
      Row(Seq(10.0), Seq(11.0), Seq(21.0)),
      Row(Seq(20.0), Seq(22.0), Seq(42.0))).u)
  }

  test(s"Reduce - sum $dtname") {
    val df = make1(Seq(1.0, 2.0).u, "x")
    val x1 = placeholder(dtype, Shape.empty) named "x_1"
    val x2 = placeholder(dtype, Shape.empty) named "x_2"
    val x = x1 + x2 named "x"
    val r = ops.reduceRows(df, x)
    assert(r === Row(3.0).u)
  }

  test(s"Simple add row - id $dtname") {
    val df = make1(Seq(1.0, 20.0).u, "in")
    val p1 = placeholder(dtype, Shape.empty) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(df, out).select("in", "out")
    compareRows(df2.collect(), Seq(Row(1.0, 1.0), Row(20.0, 20.0)).u)
  }

  test(s"Simple add - one row $dtname") {
    val df = sql.createDataFrame(Seq(
      10.0->11.0,
      20.0->22.0).u).toDF("a", "b")
    val a = placeholder(dtype, Shape.empty) named "a"
    val b = placeholder(dtype, Shape.empty) named "b"
    val out = a + b named "out"
    val df2 = ops.mapRows(df, out).select("a", "b","out")
    compareRows(df2.collect(), Seq(
      Row(10.0, 11.0, 21.0),
      Row(20.0, 22.0, 42.0)).u)
  }

  test(s"Simple add row - 1 dim $dtname") {
    val a = placeholder(dtype, Shape(1)) named "a"
    val b = placeholder(dtype, Shape(1)) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(10.0)->Seq(11.0),
      Seq(20.0)->Seq(22.0)).u).toDF("a", "b")
    val adf = ops.analyze(df)
    val df2 = ops.mapRows(adf, out).select("a", "b","out")
    compareRows(df2.collect(), Seq(
      Row(Seq(10.0), Seq(11.0), Seq(21.0)),
      Row(Seq(20.0), Seq(22.0), Seq(42.0))).u)
  }

  test(s"Simple add row - 1 dim unknown rows $dtname") {
    val a = placeholder(dtype, Shape(Unknown)) named "a"
    val b = placeholder(dtype, Shape(Unknown)) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(10.0, 10.0)->Seq(11.0, 11.0),
      Seq(20.0)->Seq(22.0)).u).toDF("a", "b")
    val adf = ops.analyze(df)
    val df2 = ops.mapRows(adf, out).select("a", "b","out")
    compareRows(df2.collect(), Seq(
      Row(Seq(10.0, 10.0), Seq(11.0, 11.0), Seq(21.0, 21.0)),
      Row(Seq(20.0), Seq(22.0), Seq(42.0))).u)
  }

  test(s"Reduce block - sum double $dtname") {
    val df = make1(Seq(1.0, 2.0).u, "x")
    val x1 = placeholder(dtype, Shape(Unknown)) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val r = ops.reduceBlocks(df, x)
    assert(r === Row(3.0).u)
  }

  test(s"Reduce block - sum double with extra column $dtname") {
    val df = sql.createDataFrame(Seq(
      ("1", 10.0),
      ("2", 11.0),
      ("3", 20.0)).u).toDF("key2", "x")
    val x1 = placeholder(dtype, Shape(Unknown)) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val r = ops.reduceBlocks(df, x)
    assert(r === Row(41.0).u)
  }

  test(s"Aggregate over rows $dtname") {
    val df = sql.createDataFrame(Seq(
      ("a", 10.0),
      ("a", 11.0),
      ("b", 20.0)).u).toDF("key", "x")
    val x1 = placeholder(dtype, Shape(Shape.Unknown)) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val df2 = ops.aggregate(df.groupBy("key"), x).select("key", "x")
    compareRows(df2.collect(), Seq(Row("a", 21.0), Row("b", 20.0)).u)
  }
}


class IntDebugSuite extends CommonOperationsSuite[Int]
  with BasicIdentityTests[Int] with BasicMonoidTests[Int] {

  override def convert(x: Double): Int = x.toInt
}

class DoubleDebugSuite extends CommonOperationsSuite[Double]
  with BasicIdentityTests[Double] with BasicMonoidTests[Double] {

  override def convert(x: Double): Double = x
}

class FloatDebugSuite extends CommonOperationsSuite[Float]
  with BasicIdentityTests[Float] with BasicMonoidTests[Float] {

  override def convert(x: Double): Float = x.toFloat
}


class LongDebugSuite extends CommonOperationsSuite[Long]
  with BasicIdentityTests[Long] with BasicMonoidTests[Long] {

  override def convert(x: Double): Long = x.toLong
}
