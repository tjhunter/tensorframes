import sbt.Keys._
import sbt._
import sbtassembly.AssemblyKeys._
import sbtassembly.AssemblyPlugin.autoImport.{ShadeRule => _, assembly => _, assemblyExcludedJars => _, assemblyOption => _, assemblyShadeRules => _}
import sbtassembly._
import sbtsparkpackage.SparkPackagePlugin.autoImport._
import sbtrelease.ReleasePlugin.autoImport._
import ReleaseTransformations._

object Shading extends Build {

  import Dependencies._

  lazy val commonSettings = Seq(
    name := "tensorframes",
    scalaVersion := sys.props.getOrElse("scala.version", "2.11.8"),
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
    test in assembly := {},
    // We only use sbt-release to update version numbers for now.
    releaseProcess := Seq[ReleaseStep](
      inquireVersions,
      setReleaseVersion,
      commitReleaseVersion,
      tagRelease,
      setNextVersion,
      commitNextVersion
    )
  )

  lazy val sparkDependencies = Seq(
    // Spark dependencies
    "org.apache.spark" %% "spark-core" % targetSparkVersion,
    "org.apache.spark" %% "spark-sql" % targetSparkVersion
  )

  // The dependencies that are platform-specific.
  lazy val allPlatformDependencies = Seq(
  )

  // The dependencies for linux only.
  // For cloud environments, it is easier to publish a smaller jar, due to limitations of spark-packages.
  lazy val linuxPlatformDependencies = Seq(
  )

  lazy val nonShadedDependencies = Seq(
    // Normal dependencies
    ModuleID("org.apache.commons", "commons-proxy", "1.0"),
    "org.scalactic" %% "scalactic" % "3.0.0",
    "org.apache.commons" % "commons-lang3" % "3.4",
    "com.typesafe.scala-logging" %% "scala-logging-api" % "2.1.2",
    "com.typesafe.scala-logging" %% "scala-logging-slf4j" % "2.1.2",
    // TensorFlow dependencies
    "org.tensorflow" % "tensorflow" % targetTensorFlowVersion
  )

  lazy val testDependencies = Seq(
    // Test dependencies
    "org.scalatest" %% "scalatest" % "3.0.0" % "test"
  )

  lazy val shadedDependencies = Seq(
    "com.google.protobuf" % "protobuf-java" % "3.5.1"
  )

  lazy val shaded = Project("shaded", file(".")).settings(
    target := target.value / "shaded",
    libraryDependencies ++= nonShadedDependencies.map(_ % "provided"),
    libraryDependencies ++= sparkDependencies.map(_ % "provided"),
    libraryDependencies ++= shadedDependencies,
    libraryDependencies ++= testDependencies,
    libraryDependencies ++= allPlatformDependencies,
    assemblyShadeRules in assembly := Seq(
      ShadeRule.rename("com.google.protobuf.**" -> "org.tensorframes.protobuf3shade.@1").inAll,
      ShadeRule.rename("google.protobuf.**" -> "org.tensorframes.google.protobuf3shade.@1").inAll
    ),
    assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
  ).settings(commonSettings: _*)

  // The artifact that is used for spark packages:
  // - includes the binary libraries, shaded protobuf
  // - does not include the other dependencies
  lazy val distribute = Project("distribution", file(".")).settings(
    target := target.value / "distribution",
    libraryDependencies := nonShadedDependencies,
    libraryDependencies ++= sparkDependencies.map(_ % "provided"),
    libraryDependencies ++= testDependencies,
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
    spAppendScalaVersion := true,
    spHomepage := "https://github.com/databricks/tensorframes",
    spShade := true,
    assembly in spPackage := (assembly in shaded).value,
    credentials += Credentials(Path.userHome / ".ssh" / "credentials_tensorframes.sbt.txt")
  ).settings(commonSettings: _*)

  // The java testing artifact: do not shade or embed anything.
  lazy val testing = Project("tfs_testing", file(".")).settings(
    target := target.value / "testing",
    libraryDependencies ++= sparkDependencies.map(_ % "provided"),
    libraryDependencies ++= nonShadedDependencies,
    libraryDependencies ++= shadedDependencies,
    libraryDependencies ++= testDependencies,
    libraryDependencies ++= allPlatformDependencies,
    // Do not attempt to run tests when building the assembly.
    test in assembly := {},
    // Spark has a dependency on protobuf2, which conflicts with protobuf3.
    // Our own dep needs to be shaded.
    assemblyShadeRules in assembly := Seq(
      ShadeRule.rename("com.google.protobuf.**" -> "org.tensorframes.protobuf3shade.@1").inAll
    ),
    assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
  ).settings(commonSettings: _*)
}
