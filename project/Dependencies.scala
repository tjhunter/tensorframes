object Dependencies {
  // The spark version
  val targetSparkVersion = sys.props.getOrElse("spark.version", "2.4.0")
  val targetTensorFlowVersion = "1.12.0"
}
