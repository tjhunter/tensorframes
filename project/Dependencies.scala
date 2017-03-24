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
  val targetSparkVersion = "2.1.0"

  val targetJCPPVersion = "1.3"

  val targetTensorFlowVersion = "1.0.1-SNAPSHOT"

  def credentialPath: File = {
    Paths.get("sbtcredentials").toAbsolutePath.toFile
  }

  // If a custom version of tensorflow is available in lib, use this one. Otherwise use the
  // default version published in maven central.
  def customTF() = {
    val baseDir = new File(".")
    val f = baseDir / "lib" / s"javacpp-$targetJCPPVersion-tensorflow-$targetTensorFlowVersion-gpu"
    if (Files.exists(f.toPath)) {
      val f2 = f.getAbsoluteFile
      println(s"Using custom tensorflow version in $f2")
      unmanagedBase := f2
    } else {
      //val vstring = s"$targetTensorFlowVersion-$targetJCPPVersion"
      val vstring = "1.0.1-1.3.3-SNAPSHOT"
      // Add other versions here if necessary
      val packages = Seq(
        "org.bytedeco.javacpp-presets" % "tensorflow" % vstring,
        "org.bytedeco.javacpp-presets" % "tensorflow" % vstring classifier "linux-x86_64"
//        "org.bytedeco.javacpp-presets" % "tensorflow" % vstring classifier "macosx-x86_64"
      )
      libraryDependencies ++= packages
    }
  }

}