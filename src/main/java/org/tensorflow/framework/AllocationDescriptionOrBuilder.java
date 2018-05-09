// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/allocation_description.proto

package org.tensorflow.framework;

public interface AllocationDescriptionOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.AllocationDescription)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional int64 requested_bytes = 1;</code>
   *
   * <pre>
   * Total number of bytes requested
   * </pre>
   */
  long getRequestedBytes();

  /**
   * <code>optional int64 allocated_bytes = 2;</code>
   *
   * <pre>
   * Total number of bytes allocated if known
   * </pre>
   */
  long getAllocatedBytes();

  /**
   * <code>optional string allocator_name = 3;</code>
   *
   * <pre>
   * Name of the allocator used
   * </pre>
   */
  java.lang.String getAllocatorName();
  /**
   * <code>optional string allocator_name = 3;</code>
   *
   * <pre>
   * Name of the allocator used
   * </pre>
   */
  com.google.protobuf.ByteString
      getAllocatorNameBytes();

  /**
   * <code>optional int64 allocation_id = 4;</code>
   *
   * <pre>
   * Identifier of the allocated buffer if known
   * </pre>
   */
  long getAllocationId();

  /**
   * <code>optional bool has_single_reference = 5;</code>
   *
   * <pre>
   * Set if this tensor only has one remaining reference
   * </pre>
   */
  boolean getHasSingleReference();

  /**
   * <code>optional uint64 ptr = 6;</code>
   *
   * <pre>
   * Address of the allocation.
   * </pre>
   */
  long getPtr();
}
