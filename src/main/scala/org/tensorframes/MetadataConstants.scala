package org.tensorframes

import org.apache.spark.sql.types.{DataType, NumericType}
import org.tensorframes.impl.{ScalarType, SupportedOperations}

/**
 * Metadata annotations that get embedded in dataframes to express tensor information.
 */
object MetadataConstants {

  /**
   * Associated with a [[Shape]] object.
   *
   * This is the shape of each of the blocks of data.
   *
   * Typically, the head of the shape is unknown because we do not know in advance how many elements
   * there will be in a block.
   */
  val shapeKey = "org.spartf.shape"

  /**
   * The string representation of a supported basic SQL type.
   *
   * The inner type of the tensor. The dataframe type for this column is going to be this type, or an array (or array
   * of arrays) of this type.
   */
  val tensorStructType = "org.sparktf.type"

  /**
   * All the SQL types supported by SparkTF.
   */
  val supportedTypes: Seq[DataType] = SupportedOperations.sqlTypes
}