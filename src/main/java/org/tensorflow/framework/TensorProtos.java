// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/tensor.proto

package org.tensorflow.framework;

public final class TensorProtos {
  private TensorProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
  }
  static com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_TensorProto_descriptor;
  static
    com.google.protobuf.GeneratedMessage.FieldAccessorTable
      internal_static_tensorflow_TensorProto_fieldAccessorTable;
  static com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_VariantTensorDataProto_descriptor;
  static
    com.google.protobuf.GeneratedMessage.FieldAccessorTable
      internal_static_tensorflow_VariantTensorDataProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n&tensorflow/core/framework/tensor.proto" +
      "\022\ntensorflow\032/tensorflow/core/framework/" +
      "resource_handle.proto\032,tensorflow/core/f" +
      "ramework/tensor_shape.proto\032%tensorflow/" +
      "core/framework/types.proto\"\214\004\n\013TensorPro" +
      "to\022#\n\005dtype\030\001 \001(\0162\024.tensorflow.DataType\022" +
      "2\n\014tensor_shape\030\002 \001(\0132\034.tensorflow.Tenso" +
      "rShapeProto\022\026\n\016version_number\030\003 \001(\005\022\026\n\016t" +
      "ensor_content\030\004 \001(\014\022\024\n\010half_val\030\r \003(\005B\002\020" +
      "\001\022\025\n\tfloat_val\030\005 \003(\002B\002\020\001\022\026\n\ndouble_val\030\006",
      " \003(\001B\002\020\001\022\023\n\007int_val\030\007 \003(\005B\002\020\001\022\022\n\nstring_" +
      "val\030\010 \003(\014\022\030\n\014scomplex_val\030\t \003(\002B\002\020\001\022\025\n\ti" +
      "nt64_val\030\n \003(\003B\002\020\001\022\024\n\010bool_val\030\013 \003(\010B\002\020\001" +
      "\022\030\n\014dcomplex_val\030\014 \003(\001B\002\020\001\022<\n\023resource_h" +
      "andle_val\030\016 \003(\0132\037.tensorflow.ResourceHan" +
      "dleProto\0227\n\013variant_val\030\017 \003(\0132\".tensorfl" +
      "ow.VariantTensorDataProto\022\026\n\nuint32_val\030" +
      "\020 \003(\rB\002\020\001\022\026\n\nuint64_val\030\021 \003(\004B\002\020\001\"g\n\026Var" +
      "iantTensorDataProto\022\021\n\ttype_name\030\001 \001(\t\022\020" +
      "\n\010metadata\030\002 \001(\014\022(\n\007tensors\030\003 \003(\0132\027.tens",
      "orflow.TensorProtoB-\n\030org.tensorflow.fra" +
      "meworkB\014TensorProtosP\001\370\001\001b\006proto3"
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
          org.tensorflow.framework.ResourceHandle.getDescriptor(),
          org.tensorflow.framework.TensorShapeProtos.getDescriptor(),
          org.tensorflow.framework.TypesProtos.getDescriptor(),
        }, assigner);
    internal_static_tensorflow_TensorProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_TensorProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessage.FieldAccessorTable(
        internal_static_tensorflow_TensorProto_descriptor,
        new java.lang.String[] { "Dtype", "TensorShape", "VersionNumber", "TensorContent", "HalfVal", "FloatVal", "DoubleVal", "IntVal", "StringVal", "ScomplexVal", "Int64Val", "BoolVal", "DcomplexVal", "ResourceHandleVal", "VariantVal", "Uint32Val", "Uint64Val", });
    internal_static_tensorflow_VariantTensorDataProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tensorflow_VariantTensorDataProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessage.FieldAccessorTable(
        internal_static_tensorflow_VariantTensorDataProto_descriptor,
        new java.lang.String[] { "TypeName", "Metadata", "Tensors", });
    org.tensorflow.framework.ResourceHandle.getDescriptor();
    org.tensorflow.framework.TensorShapeProtos.getDescriptor();
    org.tensorflow.framework.TypesProtos.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
