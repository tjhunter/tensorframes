// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/reader_base.proto

package org.tensorflow.framework;

public final class ReaderBaseProtos {
  private ReaderBaseProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
  }
  static com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_ReaderBaseState_descriptor;
  static
    com.google.protobuf.GeneratedMessage.FieldAccessorTable
      internal_static_tensorflow_ReaderBaseState_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n+tensorflow/core/framework/reader_base." +
      "proto\022\ntensorflow\"r\n\017ReaderBaseState\022\024\n\014" +
      "work_started\030\001 \001(\003\022\025\n\rwork_finished\030\002 \001(" +
      "\003\022\034\n\024num_records_produced\030\003 \001(\003\022\024\n\014curre" +
      "nt_work\030\004 \001(\014B1\n\030org.tensorflow.framewor" +
      "kB\020ReaderBaseProtosP\001\370\001\001b\006proto3"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
        }, assigner);
    internal_static_tensorflow_ReaderBaseState_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_ReaderBaseState_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessage.FieldAccessorTable(
        internal_static_tensorflow_ReaderBaseState_descriptor,
        new java.lang.String[] { "WorkStarted", "WorkFinished", "NumRecordsProduced", "CurrentWork", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}
