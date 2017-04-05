package org.tensorframes

import org.apache.spark.sql.types._


class ColumnInformation private (
      val field: StructField,
      val stf: Option[SparkTFColInfo]) extends Serializable {

  import MetadataConstants._

  def columnName: String = field.name

  def merged: StructField = {
    val b = new MetadataBuilder().withMetadata(field.metadata)
    for (info <- stf) {
      b.putLongArray(shapeKey, info.shape.dims.toArray)
      b.putString(tensorStructType, info.dataType.toString)
    }
    val meta = b.build()
    field.copy(metadata = meta)
  }
  
  override def equals(o: Any): Boolean = o match {
    case ci: ColumnInformation => ci.field == field && ci.stf == stf
    case _ => false
  }
  
  override def hashCode: Int = {
    field.hashCode * 31 + stf.hashCode()
  }
}

object ColumnInformation extends Logging {
  import MetadataConstants._
  import Shape.Unknown

  /**
   * Reads meta data info encoded in the field information. If these metadata info are missing,
   * it gets the info it can from the the structure.
   */
  def apply(field: StructField): ColumnInformation = {
    val meta = extract(field.metadata).orElse {
      // Do not support nullable for now.
      if (field.nullable) {
        // TODO switch back
//        None
        extractFromRow(field.dataType)
      } else {
        extractFromRow(field.dataType)
      }
    }
    new ColumnInformation(field, meta)
  }

  def apply(field: StructField, info: SparkTFColInfo): ColumnInformation = {
    new ColumnInformation(field, Some(info))
  }

  def apply(field: StructField, info: Option[SparkTFColInfo]): ColumnInformation = {
    new ColumnInformation(field, info)
  }

  def unapply(x: ColumnInformation): Option[(StructField, Option[SparkTFColInfo])] = {
    Some((x.field, x.stf))
  }

  /**
   * Returns a struct field with all the relevant information about shape filled out in the
   * metadata.
    *
    * @param name the name of the field
   * @param scalarType the data type
   * @param blockShape the shape of the block
   */
  def structField(name: String, scalarType: NumericType, blockShape: Shape): StructField = {
    val i = SparkTFColInfo(blockShape, scalarType)
    val f = StructField(name, sqlType(scalarType, blockShape.tail), nullable = false)
    ColumnInformation(f, i).merged
  }
  
  private def sqlType(scalarType: NumericType, shape: Shape): DataType = {
    if (shape.dims.isEmpty) {
      scalarType
    } else {
      ArrayType(sqlType(scalarType, shape.tail), containsNull = false)
    }
  }

  private def extract(meta: Metadata): Option[SparkTFColInfo] = {
    // Try to read the metadata information.
    val shape = if (meta.contains(shapeKey)) {
      Option(meta.getLongArray(shapeKey)).map(Shape.apply)
    } else {
      None
    }
    val tpe = if (meta.contains(tensorStructType)) {
      Option(meta.getString(tensorStructType)).flatMap(getType)
    } else {
      None
    }
    for {
      s <- shape
      t <- tpe
    } yield SparkTFColInfo(s, t)
  }

  private def getType(s: String): Option[NumericType] = {
    supportedTypes.find(_.toString == s)
  }

  /**
   * Tries to extract information about the type from the data type.
    *
    * @return
   */
  private def extractFromRow(dt: DataType): Option[SparkTFColInfo] = dt match {
    case x: NumericType if MetadataConstants.supportedTypes.contains(dt) =>
      logTrace("numerictype: " + x)
      // It is a basic type that we understand
      Some(SparkTFColInfo(Shape(Unknown), x))
    case x: ArrayType =>
      logTrace("arraytype: " + x)
      // Look into the array to figure out the type.
      extractFromRow(x.elementType).map { info =>
        SparkTFColInfo(info.shape.prepend(Unknown), info.dataType)
      }
    case _ =>
      logTrace("not understood: " + dt)
      // Not understood.
      None
  }
}
