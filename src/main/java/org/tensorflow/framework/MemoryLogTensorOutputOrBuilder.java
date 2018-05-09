// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/log_memory.proto

package org.tensorflow.framework;

public interface MemoryLogTensorOutputOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.MemoryLogTensorOutput)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional int64 step_id = 1;</code>
   *
   * <pre>
   * Process-unique step id.
   * </pre>
   */
  long getStepId();

  /**
   * <code>optional string kernel_name = 2;</code>
   *
   * <pre>
   * Name of the kernel producing an output as set in GraphDef, e.g.,
   * "affine2/weights/Assign".
   * </pre>
   */
  java.lang.String getKernelName();
  /**
   * <code>optional string kernel_name = 2;</code>
   *
   * <pre>
   * Name of the kernel producing an output as set in GraphDef, e.g.,
   * "affine2/weights/Assign".
   * </pre>
   */
  com.google.protobuf.ByteString
      getKernelNameBytes();

  /**
   * <code>optional int32 index = 3;</code>
   *
   * <pre>
   * Index of the output being set.
   * </pre>
   */
  int getIndex();

  /**
   * <code>optional .tensorflow.TensorDescription tensor = 4;</code>
   *
   * <pre>
   * Output tensor details.
   * </pre>
   */
  boolean hasTensor();
  /**
   * <code>optional .tensorflow.TensorDescription tensor = 4;</code>
   *
   * <pre>
   * Output tensor details.
   * </pre>
   */
  org.tensorflow.framework.TensorDescription getTensor();
  /**
   * <code>optional .tensorflow.TensorDescription tensor = 4;</code>
   *
   * <pre>
   * Output tensor details.
   * </pre>
   */
  org.tensorflow.framework.TensorDescriptionOrBuilder getTensorOrBuilder();
}
