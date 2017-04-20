package org.tensorframes

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, DataType, NumericType}

import org.tensorframes.impl.{ScalarType, SupportedOperations}

/**
  * Some useful methods for operating on dataframes that are not part of the official API (and thus may change anytime).
 */
trait ExperimentalOperations {

  /**
   * Takes each block and converts it to a single row in a dataframe by augmenting the dimension of each column by
   * one element.
    *
    * @param df the input dataframe.
   * @return
   */
  def convertBlockToRow(df: DataFrame): DataFrame = ???

  def explainDetailed(df: DataFrame): DataFrameInfo = DataFrameInfo.get(df)

  /**
   * Checks that each row of the dataframe has a type that is compatible with SparkTF.
   *
   * The returned dataframe embeds metadata information about the shape of the elements.
   *
   * This information is also checked as runtime, so it is mostly convenient for debugging.
    *
    * @param df
   * @return
   */
  def analyze(df: DataFrame): DataFrame = {
    // Get the schema:
    val dfInfo = ExtraOperations.deepAnalyzeDataFrame(df)

    // Add the metadata to the columns.
    val cols = dfInfo.cols.map {
      case ColumnInformation(f, None) => col(f.name)
      case ci @ ColumnInformation(f, Some(info)) =>
        // We have some extra metadata to add to this column
        col(f.name).as(f.name, ci.merged.metadata)
    }
    df.select(cols: _*)
  }
}

private[tensorframes] object ExtraOperations extends ExperimentalOperations with Logging {
  import Shape.Unknown

  // **** Deep partition analysis *****

  /**
   * Performs a deep analysis of every single element of the dataframe to
   * determine if:
   *  - each element is of a known type
   *  - the (tensorial) shapes of each element are compatible within each column.
   *    For example, if a column contains vectors, it checks that all the vectors have the
   *    same size.
    *
    * @param df
   * @return
   */
  // TODO(tjh) add support for Spark's VectorUDT and MatrixUDT
  // TODO(tjh) add test when the number of partitions is greater than the number of elements
  def deepAnalyzeDataFrame(df: DataFrame): DataFrameInfo = {
    val numCols = df.schema.fields.length

    // Some partitions may be empty, so make sure they do not pollute the analysis.
    val rowPartitionInfo: Array[Option[Array[Option[Shape]]]] = df.rdd.mapPartitions { it =>
      var partitionSize = 0
      val opt1 = it.map { row =>
        partitionSize += 1
        val analyzed = (0 until row.size).map(i => analyzeData(row.get(i)))
        logDebug(s"analyzed: $analyzed")
        analyzed.toArray
      } .reduceOption(ExtraOperations.f2)
      logDebug(s"opt1: $opt1")

      val it2 = opt1.getOrElse(Array.fill(numCols)(None))
      logDebug(s"it2: ${it2.toSeq}")

      // Add the size of the partition as the first dimension.
      val it3: Array[Option[Shape]] = it2.map {
        case None => None
        case Some(shape) => Some(shape.prepend(partitionSize))
      }
      val res = if (partitionSize == 0) { None } else { Some(it3) }
      Iterable(res).toIterator
    } .collect()

    // Remove the empty partitions
    val partitionInfo: Array[Array[Option[Shape]]] = rowPartitionInfo.flatten

    val agg = partitionInfo.reduceOption(ExtraOperations.f2).getOrElse {
      // The dataframe is empty, there is nothing extra we can say about that.
      Array.fill(numCols)(None)
    }

    val allInfo = agg.zip(df.schema.fields).map { case (o, f) =>
      val i = for {
        shape <- o
        tensorType <- extractBasicType(f.dataType)
      } yield SparkTFColInfo(shape, tensorType)
      ColumnInformation(f, i)
    }

    DataFrameInfo(allInfo)
  }

  private def extractBasicType(dt: DataType): Option[ScalarType] = dt match {
    case x: NumericType => Some(SupportedOperations.opsFor(x).scalarType)
    case x: ArrayType => extractBasicType(x.elementType)
    case _ => None
  }

  def analyzeData(x: Any): Option[Shape] = x match {
    case null => None
    case u: Array[_] =>
      val shapes = u.map(analyzeData)
      mergeStructs(shapes).map(_.prepend(u.length))
    case u: Seq[_] =>
      val shapes = u.map(analyzeData)
      mergeStructs(shapes).map(_.prepend(u.size))
    case z if SupportedOperations.hasOps(z) => Some(Shape.empty)
    case _ =>
      None
//      throw new Exception(s"Type of item '$x' (${x.getClass.getSimpleName}) is not understood")
  }

  private def f(o1: Option[Shape], o2: Option[Shape]): Option[Shape] =
    for (s1 <- o1 ; s2 <- o2; s <- merge(s1, s2)) yield s

  def f2(os1: Array[Option[Shape]], os2: Array[Option[Shape]]) =
    os1.zip(os2).map(z => f(z._1, z._2))

  private def mergeStructs(shapes: Seq[Option[Shape]]): Option[Shape] = {
    if (shapes.isEmpty) {
      Some(Shape.empty)
    } else {
      shapes.reduceOption(f).flatten
    }
  }

  private def merge(shape1: Shape, shape2: Shape): Option[Shape] = {
    if (shape1.dims.length != shape2.dims.length) {
      None
    } else {
      val dims = shape1.dims.zip(shape2.dims).map {
        case (x1, x2) if x1 == x2 => x1
        case _ => Unknown.toLong
      }
      Some(Shape(dims.toArray))
    }
  }

  // **** Other operations ****
}
