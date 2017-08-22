import sbt._
import sbtsparkpackage.SparkPackagePlugin
import sbtsparkpackage.SparkPackagePlugin.{autoImport => sp}
import Keys._

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.Locale

import xml.{NodeSeq, Node => XNode, Elem}
import xml.transform.{RuleTransformer, RewriteRule}

object Dependencies {
  // The spark version
  val targetSparkVersion = sys.props.getOrElse("spark.version", "2.1.1")

  val targetTensorFlowVersion = "1.3.0"

}