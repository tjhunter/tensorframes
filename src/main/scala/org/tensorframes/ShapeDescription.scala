package org.tensorframes

/**
 * In order to go around a limitation of the C++ kernel of TensorFlow, the shape of the outputs
 * must be provided to the interface.
 *
 * It also includes the names of the fetches (since they may be used in some other ways within
 * the graph, and not all of them may be requested).
 */
// TODO: rename to TFExtraInfo (?)
// Some data constants to be used to fill the placeholders
case class ShapeDescription(
    out: Map[String, Shape],
    requestedFetches: Seq[String],
    inputs: Map[NodePath, FieldName]) {
}

object ShapeDescription {
  val empty = ShapeDescription(Map.empty, Seq.empty, Map.empty)
}