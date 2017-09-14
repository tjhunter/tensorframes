package org.tensorframes.dsl

import java.io.{BufferedReader, InputStreamReader, File}
import java.nio.file.Files
import java.nio.charset.StandardCharsets
import org.tensorframes.Logging
import org.scalatest.Matchers

import scala.collection.JavaConverters._

object ExtractNodes extends Matchers with Logging {

  def executeCommand(py: String): Map[String, String] = {
    val content =
      s"""
         |from __future__ import print_function
         |import tensorflow as tf
         |
         |$py
         |g = tf.get_default_graph().as_graph_def()
         |for n in g.node:
         |    print(">>>>>", str(n.name), "<<<<<<")
         |    print(n)
       """.stripMargin
    val f = File.createTempFile("pythonTest", ".py")
    logTrace(s"Created temp file ${f.getAbsolutePath}")
    Files.write(f.toPath, content.getBytes(StandardCharsets.UTF_8))
    // Using the standard python installation in the PATH. It needs to have TensorFlow installed.
    val p = new ProcessBuilder("python", f.getAbsolutePath).start()
    val s = p.getInputStream
    val isr = new InputStreamReader(s)
    val br = new BufferedReader(isr)
    var res: String = ""
    var str: String = ""
    while(str != null) {
      str = br.readLine()
      if (str != null) {
        res = res + "\n" + str
      }
    }

    p.waitFor()
    assert(p.exitValue() === 0, (p.exitValue(),
      {
        println(content)
        s"===========\n$content\n==========="
      }))
    res.split(">>>>>").map(_.trim).filterNot(_.isEmpty).map { b =>
      val zs = b.split("\n")
      val node = zs.head.dropRight(7)
      val rest = zs.tail
      node -> rest.mkString("\n")
    } .toMap
  }

  def compareOutput(py: String, nodes: Operation*): Unit = {
    val g = TestUtilities.buildGraph(nodes.head, nodes.tail:_*)
    val m1 = g.getNodeList.asScala.map { n =>
      n.getName -> n.toString.trim
    } .toMap
    val pym = executeCommand(py)
    logTrace(s"m1 = '$m1'")
    logTrace(s"pym = '$pym'")
    assert((m1.keySet -- pym.keySet).isEmpty, {
      val diff = (m1.keySet -- pym.keySet).toSeq.sorted
      s"Found extra nodes in scala: $diff"
    })
    assert((pym.keySet -- m1.keySet).isEmpty, {
      val diff = (pym.keySet -- m1.keySet).toSeq.sorted
      s"Found extra nodes in python: $diff"
    })
    for (k <- m1.keySet) {
      assert(m1(k) === pym(k),
        s"scala=${m1(k)}\npython=${pym(k)}")
    }
  }
}
