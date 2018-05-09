// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/step_stats.proto

package org.tensorflow.framework;

public interface NodeExecStatsOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.NodeExecStats)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional string node_name = 1;</code>
   *
   * <pre>
   * TODO(tucker): Use some more compact form of node identity than
   * the full string name.  Either all processes should agree on a
   * global id (cost_id?) for each node, or we should use a hash of
   * the name.
   * </pre>
   */
  java.lang.String getNodeName();
  /**
   * <code>optional string node_name = 1;</code>
   *
   * <pre>
   * TODO(tucker): Use some more compact form of node identity than
   * the full string name.  Either all processes should agree on a
   * global id (cost_id?) for each node, or we should use a hash of
   * the name.
   * </pre>
   */
  com.google.protobuf.ByteString
      getNodeNameBytes();

  /**
   * <code>optional int64 all_start_micros = 2;</code>
   */
  long getAllStartMicros();

  /**
   * <code>optional int64 op_start_rel_micros = 3;</code>
   */
  long getOpStartRelMicros();

  /**
   * <code>optional int64 op_end_rel_micros = 4;</code>
   */
  long getOpEndRelMicros();

  /**
   * <code>optional int64 all_end_rel_micros = 5;</code>
   */
  long getAllEndRelMicros();

  /**
   * <code>repeated .tensorflow.AllocatorMemoryUsed memory = 6;</code>
   */
  java.util.List<org.tensorflow.framework.AllocatorMemoryUsed> 
      getMemoryList();
  /**
   * <code>repeated .tensorflow.AllocatorMemoryUsed memory = 6;</code>
   */
  org.tensorflow.framework.AllocatorMemoryUsed getMemory(int index);
  /**
   * <code>repeated .tensorflow.AllocatorMemoryUsed memory = 6;</code>
   */
  int getMemoryCount();
  /**
   * <code>repeated .tensorflow.AllocatorMemoryUsed memory = 6;</code>
   */
  java.util.List<? extends org.tensorflow.framework.AllocatorMemoryUsedOrBuilder> 
      getMemoryOrBuilderList();
  /**
   * <code>repeated .tensorflow.AllocatorMemoryUsed memory = 6;</code>
   */
  org.tensorflow.framework.AllocatorMemoryUsedOrBuilder getMemoryOrBuilder(
      int index);

  /**
   * <code>repeated .tensorflow.NodeOutput output = 7;</code>
   */
  java.util.List<org.tensorflow.framework.NodeOutput> 
      getOutputList();
  /**
   * <code>repeated .tensorflow.NodeOutput output = 7;</code>
   */
  org.tensorflow.framework.NodeOutput getOutput(int index);
  /**
   * <code>repeated .tensorflow.NodeOutput output = 7;</code>
   */
  int getOutputCount();
  /**
   * <code>repeated .tensorflow.NodeOutput output = 7;</code>
   */
  java.util.List<? extends org.tensorflow.framework.NodeOutputOrBuilder> 
      getOutputOrBuilderList();
  /**
   * <code>repeated .tensorflow.NodeOutput output = 7;</code>
   */
  org.tensorflow.framework.NodeOutputOrBuilder getOutputOrBuilder(
      int index);

  /**
   * <code>optional string timeline_label = 8;</code>
   */
  java.lang.String getTimelineLabel();
  /**
   * <code>optional string timeline_label = 8;</code>
   */
  com.google.protobuf.ByteString
      getTimelineLabelBytes();

  /**
   * <code>optional int64 scheduled_micros = 9;</code>
   */
  long getScheduledMicros();

  /**
   * <code>optional uint32 thread_id = 10;</code>
   */
  int getThreadId();

  /**
   * <code>repeated .tensorflow.AllocationDescription referenced_tensor = 11;</code>
   */
  java.util.List<org.tensorflow.framework.AllocationDescription> 
      getReferencedTensorList();
  /**
   * <code>repeated .tensorflow.AllocationDescription referenced_tensor = 11;</code>
   */
  org.tensorflow.framework.AllocationDescription getReferencedTensor(int index);
  /**
   * <code>repeated .tensorflow.AllocationDescription referenced_tensor = 11;</code>
   */
  int getReferencedTensorCount();
  /**
   * <code>repeated .tensorflow.AllocationDescription referenced_tensor = 11;</code>
   */
  java.util.List<? extends org.tensorflow.framework.AllocationDescriptionOrBuilder> 
      getReferencedTensorOrBuilderList();
  /**
   * <code>repeated .tensorflow.AllocationDescription referenced_tensor = 11;</code>
   */
  org.tensorflow.framework.AllocationDescriptionOrBuilder getReferencedTensorOrBuilder(
      int index);

  /**
   * <code>optional .tensorflow.MemoryStats memory_stats = 12;</code>
   */
  boolean hasMemoryStats();
  /**
   * <code>optional .tensorflow.MemoryStats memory_stats = 12;</code>
   */
  org.tensorflow.framework.MemoryStats getMemoryStats();
  /**
   * <code>optional .tensorflow.MemoryStats memory_stats = 12;</code>
   */
  org.tensorflow.framework.MemoryStatsOrBuilder getMemoryStatsOrBuilder();
}
