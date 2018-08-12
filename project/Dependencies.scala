object Dependencies {
  // The spark version
  val targetSparkVersion = sys.props.getOrElse("spark.version", "2.3.1")
  val targetTensorFlowVersion = "1.10.0-rc1"
}
