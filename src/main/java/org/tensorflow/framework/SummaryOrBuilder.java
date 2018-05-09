// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/summary.proto

package org.tensorflow.framework;

public interface SummaryOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.Summary)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   *
   * <pre>
   * Set of values for the summary.
   * </pre>
   */
  java.util.List<org.tensorflow.framework.Summary.Value> 
      getValueList();
  /**
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   *
   * <pre>
   * Set of values for the summary.
   * </pre>
   */
  org.tensorflow.framework.Summary.Value getValue(int index);
  /**
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   *
   * <pre>
   * Set of values for the summary.
   * </pre>
   */
  int getValueCount();
  /**
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   *
   * <pre>
   * Set of values for the summary.
   * </pre>
   */
  java.util.List<? extends org.tensorflow.framework.Summary.ValueOrBuilder> 
      getValueOrBuilderList();
  /**
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   *
   * <pre>
   * Set of values for the summary.
   * </pre>
   */
  org.tensorflow.framework.Summary.ValueOrBuilder getValueOrBuilder(
      int index);
}
