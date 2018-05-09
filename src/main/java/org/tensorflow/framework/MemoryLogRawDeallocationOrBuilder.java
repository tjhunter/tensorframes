// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/log_memory.proto

package org.tensorflow.framework;

public interface MemoryLogRawDeallocationOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.MemoryLogRawDeallocation)
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
   * <code>optional string operation = 2;</code>
   *
   * <pre>
   * Name of the operation making the deallocation.
   * </pre>
   */
  java.lang.String getOperation();
  /**
   * <code>optional string operation = 2;</code>
   *
   * <pre>
   * Name of the operation making the deallocation.
   * </pre>
   */
  com.google.protobuf.ByteString
      getOperationBytes();

  /**
   * <code>optional int64 allocation_id = 3;</code>
   *
   * <pre>
   * Id of the tensor buffer being deallocated, used to match to a
   * corresponding allocation.
   * </pre>
   */
  long getAllocationId();

  /**
   * <code>optional string allocator_name = 4;</code>
   *
   * <pre>
   * Name of the allocator used.
   * </pre>
   */
  java.lang.String getAllocatorName();
  /**
   * <code>optional string allocator_name = 4;</code>
   *
   * <pre>
   * Name of the allocator used.
   * </pre>
   */
  com.google.protobuf.ByteString
      getAllocatorNameBytes();

  /**
   * <code>optional bool deferred = 5;</code>
   *
   * <pre>
   * True if the deallocation is queued and will be performed later,
   * e.g. for GPU lazy freeing of buffers.
   * </pre>
   */
  boolean getDeferred();
}
