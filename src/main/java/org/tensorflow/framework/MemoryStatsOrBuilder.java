// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/step_stats.proto

package org.tensorflow.framework;

public interface MemoryStatsOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.MemoryStats)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional int64 temp_memory_size = 1;</code>
   */
  long getTempMemorySize();

  /**
   * <code>optional int64 persistent_memory_size = 3;</code>
   */
  long getPersistentMemorySize();

  /**
   * <code>repeated int64 persistent_tensor_alloc_ids = 5;</code>
   */
  java.util.List<java.lang.Long> getPersistentTensorAllocIdsList();
  /**
   * <code>repeated int64 persistent_tensor_alloc_ids = 5;</code>
   */
  int getPersistentTensorAllocIdsCount();
  /**
   * <code>repeated int64 persistent_tensor_alloc_ids = 5;</code>
   */
  long getPersistentTensorAllocIds(int index);

  /**
   * <code>optional int64 device_temp_memory_size = 2 [deprecated = true];</code>
   */
  @java.lang.Deprecated long getDeviceTempMemorySize();

  /**
   * <code>optional int64 device_persistent_memory_size = 4 [deprecated = true];</code>
   */
  @java.lang.Deprecated long getDevicePersistentMemorySize();

  /**
   * <code>repeated int64 device_persistent_tensor_alloc_ids = 6 [deprecated = true];</code>
   */
  @java.lang.Deprecated java.util.List<java.lang.Long> getDevicePersistentTensorAllocIdsList();
  /**
   * <code>repeated int64 device_persistent_tensor_alloc_ids = 6 [deprecated = true];</code>
   */
  @java.lang.Deprecated int getDevicePersistentTensorAllocIdsCount();
  /**
   * <code>repeated int64 device_persistent_tensor_alloc_ids = 6 [deprecated = true];</code>
   */
  @java.lang.Deprecated long getDevicePersistentTensorAllocIds(int index);
}
