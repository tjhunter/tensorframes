// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/function.proto

package org.tensorflow.framework;

public interface FunctionDefOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.FunctionDef)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional .tensorflow.OpDef signature = 1;</code>
   *
   * <pre>
   * The definition of the function's name, arguments, return values,
   * attrs etc.
   * </pre>
   */
  boolean hasSignature();
  /**
   * <code>optional .tensorflow.OpDef signature = 1;</code>
   *
   * <pre>
   * The definition of the function's name, arguments, return values,
   * attrs etc.
   * </pre>
   */
  org.tensorflow.framework.OpDef getSignature();
  /**
   * <code>optional .tensorflow.OpDef signature = 1;</code>
   *
   * <pre>
   * The definition of the function's name, arguments, return values,
   * attrs etc.
   * </pre>
   */
  org.tensorflow.framework.OpDefOrBuilder getSignatureOrBuilder();

  /**
   * <code>map&lt;string, .tensorflow.AttrValue&gt; attr = 5;</code>
   *
   * <pre>
   * Attributes specific to this function definition.
   * </pre>
   */
  java.util.Map<java.lang.String, org.tensorflow.framework.AttrValue>
  getAttr();

  /**
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   *
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   */
  java.util.List<org.tensorflow.framework.NodeDef> 
      getNodeDefList();
  /**
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   *
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   */
  org.tensorflow.framework.NodeDef getNodeDef(int index);
  /**
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   *
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   */
  int getNodeDefCount();
  /**
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   *
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   */
  java.util.List<? extends org.tensorflow.framework.NodeDefOrBuilder> 
      getNodeDefOrBuilderList();
  /**
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   *
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   */
  org.tensorflow.framework.NodeDefOrBuilder getNodeDefOrBuilder(
      int index);

  /**
   * <code>map&lt;string, string&gt; ret = 4;</code>
   *
   * <pre>
   * A mapping from the output arg names from `signature` to the
   * outputs from `node_def` that should be returned by the function.
   * </pre>
   */
  java.util.Map<java.lang.String, java.lang.String>
  getRet();
}
