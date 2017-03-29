import sbt._
import sbt.Keys._
import sbtsparkpackage.SparkPackagePlugin.autoImport._
import sbtassembly._
import sbtassembly.AssemblyKeys._
import sbtassembly.AssemblyPlugin.autoImport.{ShadeRule => _, assembly => _, assemblyExcludedJars => _, assemblyOption => _, assemblyShadeRules => _, _}

object Shading extends Build {

  import Dependencies._


  lazy val commonSettings = Seq(
    version := "0.2.5",
    name := "tensorframes",
    scalaVersion := "2.11.8",
    organization := "databricks",
    sparkVersion := targetSparkVersion,
    licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")),
    // System conf
    parallelExecution := false,
    javaOptions in run += "-Xmx6G",
    // Add all the python files in the final binary
    unmanagedResourceDirectories in Compile += {
      baseDirectory.value / "src/main/python/"
    },
    // Spark packages does not like this part
    test in assembly := {}

  )

  lazy val sparkDependencies = Seq(
    // Spark dependencies
    "org.apache.spark" %% "spark-core" % targetSparkVersion,
    "org.apache.spark" %% "spark-sql" % targetSparkVersion
  )

  lazy val nonShadedDependencies = Seq(
    // Normal dependencies
    ModuleID("org.apache.commons", "commons-proxy", "1.0"),
    "org.scalactic" %% "scalactic" % "3.0.0",
    "org.apache.commons" % "commons-lang3" % "3.4",
    "com.typesafe.scala-logging" %% "scala-logging-api" % "2.1.2",
    "com.typesafe.scala-logging" %% "scala-logging-slf4j" % "2.1.2",
    // TensorFlow dependencies
    "org.bytedeco" % "javacpp" % targetJCPPVersion,
    "org.bytedeco.javacpp-presets" % "tensorflow" % targetTensorFlowVersion,
    "org.bytedeco.javacpp-presets" % "tensorflow" % targetTensorFlowVersion classifier "linux-x86_64",
    "org.bytedeco.javacpp-presets" % "tensorflow" % targetTensorFlowVersion classifier "macosx-x86_64"
  )

  lazy val testDependencies = Seq(
    // Test dependencies
    "org.scalatest" %% "scalatest" % "3.0.0" % "test"
  )

  lazy val shadedDependencies = Seq(
    "com.google.protobuf" % "protobuf-java" % "3.2.0"
  )

  lazy val shaded = Project("shaded", file(".")).settings(
    libraryDependencies ++= nonShadedDependencies.map(_ % "provided"),
    libraryDependencies ++= sparkDependencies.map(_ % "provided"),
    libraryDependencies ++= shadedDependencies,
    libraryDependencies ++= testDependencies,
    target := target.value / "shaded",
    assemblyShadeRules in assembly := Seq(
      ShadeRule.rename("com.google.protobuf.**" -> "org.tensorframes.protobuf3shade.@1").inAll
    )
  ).settings(commonSettings: _*)

  lazy val distribute = Project("distribution", file(".")).settings(
    spName := "databricks/tensorframes",
    spShortDescription := "TensorFlow wrapper for DataFrames on Apache Spark",
    spDescription := {
        """TensorFrames (TensorFlow on Spark DataFrames) lets you manipulate Spark's DataFrames with
          | TensorFlow programs.
          |
          |This package provides a small runtime to express and run TensorFlow computation graphs.
          |TensorFlow programs can be interpreted from:
          | - the official Python API
          | - the semi-official protocol buffer graph description format
          | - the Scala DSL embedded with TensorFrames (experimental)
          |
          |For more information, visit the TensorFrames user guide:
          |
        """.stripMargin
      },
    target := target.value / "distribution",
    spShade := true,
    assembly in spPackage := (assembly in shaded).value,
    assemblyOption in spPackage := (assemblyOption in assembly).value.copy(includeScala = false),
    libraryDependencies := nonShadedDependencies,
    libraryDependencies ++= sparkDependencies.map(_ % "provided"),
    libraryDependencies ++= testDependencies
  ).settings(commonSettings: _*)

  // TODO: move to the shaded assembly
  lazy val customAssembly = Project("custom-assembly", file(".")).settings(
    assemblyExcludedJars in assembly := {
      val cp = (fullClasspath in assembly).value
      val excludes = Set(
        "tensorflow-sources.jar",
        "tensorflow-javadoc.jar",
        "tensorflow-1.0.0-1.3-macosx-x86_64.jar" // This is not the main target, excluding
      )
      cp filter { s => excludes.contains(s.data.getName) }
    },
    // Spark has a dependency on protobuf2, which conflicts with protobuf3.
    // Our own dep needs to be shaded.
    assemblyShadeRules in assembly := Seq(
      ShadeRule.rename("com.google.protobuf.**" -> "org.tensorframes.protobuf3shade.@1").inAll
    ),
    assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
  )
}