// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/graph.proto

package org.tensorflow.framework;

public final class GraphProtos {
  private GraphProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
  }
  static com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_GraphDef_descriptor;
  static
    com.google.protobuf.GeneratedMessage.FieldAccessorTable
      internal_static_tensorflow_GraphDef_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n%tensorflow/core/framework/graph.proto\022" +
      "\ntensorflow\032(tensorflow/core/framework/n" +
      "ode_def.proto\032(tensorflow/core/framework" +
      "/function.proto\032(tensorflow/core/framewo" +
      "rk/versions.proto\"\235\001\n\010GraphDef\022!\n\004node\030\001" +
      " \003(\0132\023.tensorflow.NodeDef\022(\n\010versions\030\004 " +
      "\001(\0132\026.tensorflow.VersionDef\022\023\n\007version\030\003" +
      " \001(\005B\002\030\001\022/\n\007library\030\002 \001(\0132\036.tensorflow.F" +
      "unctionDefLibraryB,\n\030org.tensorflow.fram" +
      "eworkB\013GraphProtosP\001\370\001\001b\006proto3"
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
          org.tensorflow.framework.NodeProto.getDescriptor(),
          org.tensorflow.framework.FunctionProtos.getDescriptor(),
          org.tensorflow.framework.VersionsProtos.getDescriptor(),
        }, assigner);
    internal_static_tensorflow_GraphDef_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_GraphDef_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessage.FieldAccessorTable(
        internal_static_tensorflow_GraphDef_descriptor,
        new java.lang.String[] { "Node", "Versions", "Version", "Library", });
    org.tensorflow.framework.NodeProto.getDescriptor();
    org.tensorflow.framework.FunctionProtos.getDescriptor();
    org.tensorflow.framework.VersionsProtos.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
