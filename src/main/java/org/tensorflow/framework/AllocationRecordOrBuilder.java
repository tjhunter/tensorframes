// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/step_stats.proto

package org.tensorflow.framework;

public interface AllocationRecordOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.AllocationRecord)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional int64 alloc_micros = 1;</code>
   *
   * <pre>
   * The timestamp of the operation.
   * </pre>
   */
  long getAllocMicros();

  /**
   * <code>optional int64 alloc_bytes = 2;</code>
   *
   * <pre>
   * Number of bytes allocated, or de-allocated if negative.
   * </pre>
   */
  long getAllocBytes();
}
