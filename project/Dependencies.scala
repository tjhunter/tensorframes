object Dependencies {
  // The spark version
  val targetSparkVersion = sys.props.getOrElse("spark.version", "2.4.3")
  val targetTensorFlowVersion = "1.13.1"
}
