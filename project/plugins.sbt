resolvers += "Spark Packages repo" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.spark-packages" %% "sbt-spark-package" % "0.2.5")

addSbtPlugin("com.github.mpeltonen" % "sbt-idea" % "1.6.0")

// You need protoc3 for this to work
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.6.3")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.3")

addSbtPlugin("com.github.gseitz" % "sbt-release" % "1.0.8")
