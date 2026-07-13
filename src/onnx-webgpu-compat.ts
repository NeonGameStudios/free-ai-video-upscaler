import * as protobuf from 'protobufjs/light';

const root = new protobuf.Root();

function defineType(
  name: string,
  fields: Array<[string, number, string, 'optional' | 'repeated', boolean?]>
): protobuf.Type {
  const type = new protobuf.Type(name);
  for (const [fieldName, id, fieldType, rule, packed] of fields) {
    type.add(new protobuf.Field(
      fieldName,
      id,
      fieldType,
      rule,
      undefined,
      packed ? { packed: true } : undefined
    ));
  }
  root.add(type);
  return type;
}

defineType('StringStringEntryProto', [
  ['key', 1, 'string', 'optional'],
  ['value', 2, 'string', 'optional'],
]);

defineType('OperatorSetIdProto', [
  ['domain', 1, 'string', 'optional'],
  ['version', 2, 'int64', 'optional'],
]);

defineType('TensorProtoSegment', [
  ['begin', 1, 'int64', 'optional'],
  ['end', 2, 'int64', 'optional'],
]);

defineType('TensorShapeProtoDimension', [
  ['dim_value', 1, 'int64', 'optional'],
  ['dim_param', 2, 'string', 'optional'],
  ['denotation', 3, 'string', 'optional'],
]);

defineType('TensorShapeProto', [
  ['dim', 1, 'TensorShapeProtoDimension', 'repeated'],
]);

defineType('TypeProtoTensor', [
  ['elem_type', 1, 'int32', 'optional'],
  ['shape', 2, 'TensorShapeProto', 'optional'],
]);

defineType('TypeProtoSequence', [
  ['elem_type', 1, 'TypeProto', 'optional'],
]);

defineType('TypeProtoMap', [
  ['key_type', 1, 'int32', 'optional'],
  ['value_type', 2, 'TypeProto', 'optional'],
]);

defineType('TypeProtoOptional', [
  ['elem_type', 1, 'TypeProto', 'optional'],
]);

defineType('TypeProtoSparseTensor', [
  ['elem_type', 1, 'int32', 'optional'],
  ['shape', 2, 'TensorShapeProto', 'optional'],
]);

defineType('TypeProto', [
  ['tensor_type', 1, 'TypeProtoTensor', 'optional'],
  ['sequence_type', 4, 'TypeProtoSequence', 'optional'],
  ['map_type', 5, 'TypeProtoMap', 'optional'],
  ['optional_type', 9, 'TypeProtoOptional', 'optional'],
  ['sparse_tensor_type', 8, 'TypeProtoSparseTensor', 'optional'],
  ['denotation', 6, 'string', 'optional'],
]);

defineType('ValueInfoProto', [
  ['name', 1, 'string', 'optional'],
  ['type', 2, 'TypeProto', 'optional'],
  ['doc_string', 3, 'string', 'optional'],
]);

defineType('TensorProto', [
  ['dims', 1, 'int64', 'repeated', true],
  ['data_type', 2, 'int32', 'optional'],
  ['segment', 3, 'TensorProtoSegment', 'optional'],
  ['float_data', 4, 'float', 'repeated', true],
  ['int32_data', 5, 'int32', 'repeated', true],
  ['string_data', 6, 'bytes', 'repeated'],
  ['int64_data', 7, 'int64', 'repeated', true],
  ['name', 8, 'string', 'optional'],
  ['raw_data', 9, 'bytes', 'optional'],
  ['double_data', 10, 'double', 'repeated', true],
  ['uint64_data', 11, 'uint64', 'repeated', true],
  ['doc_string', 12, 'string', 'optional'],
  ['external_data', 13, 'StringStringEntryProto', 'repeated'],
  ['data_location', 14, 'int32', 'optional'],
  ['metadata_props', 16, 'StringStringEntryProto', 'repeated'],
]);

defineType('AttributeProto', [
  ['name', 1, 'string', 'optional'],
  ['f', 2, 'float', 'optional'],
  ['i', 3, 'int64', 'optional'],
  ['s', 4, 'bytes', 'optional'],
  ['t', 5, 'TensorProto', 'optional'],
  ['g', 6, 'GraphProto', 'optional'],
  ['floats', 7, 'float', 'repeated', true],
  ['ints', 8, 'int64', 'repeated', true],
  ['strings', 9, 'bytes', 'repeated'],
  ['tensors', 10, 'TensorProto', 'repeated'],
  ['graphs', 11, 'GraphProto', 'repeated'],
  ['type', 20, 'int32', 'optional'],
]);

defineType('NodeProto', [
  ['input', 1, 'string', 'repeated'],
  ['output', 2, 'string', 'repeated'],
  ['name', 3, 'string', 'optional'],
  ['op_type', 4, 'string', 'optional'],
  ['attribute', 5, 'AttributeProto', 'repeated'],
  ['doc_string', 6, 'string', 'optional'],
  ['domain', 7, 'string', 'optional'],
]);

defineType('SparseTensorProto', [
  ['values', 1, 'TensorProto', 'optional'],
  ['indices', 2, 'TensorProto', 'optional'],
  ['dims', 3, 'uint64', 'repeated', true],
]);

defineType('TensorAnnotation', [
  ['tensor_name', 1, 'string', 'optional'],
  ['quant_parameter_tensor_names', 2, 'StringStringEntryProto', 'repeated'],
]);

defineType('GraphProto', [
  ['node', 1, 'NodeProto', 'repeated'],
  ['name', 2, 'string', 'optional'],
  ['initializer', 5, 'TensorProto', 'repeated'],
  ['doc_string', 10, 'string', 'optional'],
  ['input', 11, 'ValueInfoProto', 'repeated'],
  ['output', 12, 'ValueInfoProto', 'repeated'],
  ['value_info', 13, 'ValueInfoProto', 'repeated'],
  ['quantization_annotation', 14, 'TensorAnnotation', 'repeated'],
  ['sparse_initializer', 15, 'SparseTensorProto', 'repeated'],
]);

defineType('TrainingInfoProto', [
  ['initialization', 1, 'GraphProto', 'optional'],
  ['algorithm', 2, 'GraphProto', 'optional'],
]);

defineType('FunctionProto', [
  ['name', 1, 'string', 'optional'],
  ['input', 2, 'string', 'repeated'],
  ['output', 3, 'string', 'repeated'],
  ['node', 4, 'NodeProto', 'repeated'],
  ['initializer', 5, 'TensorProto', 'repeated'],
  ['doc_string', 6, 'string', 'optional'],
  ['opset_import', 7, 'OperatorSetIdProto', 'repeated'],
  ['domain', 8, 'string', 'optional'],
  ['attribute', 9, 'string', 'repeated'],
  ['attribute_proto', 10, 'AttributeProto', 'repeated'],
  ['value_info', 11, 'ValueInfoProto', 'repeated'],
]);

const modelType = defineType('ModelProto', [
  ['ir_version', 1, 'int64', 'optional'],
  ['producer_name', 2, 'string', 'optional'],
  ['producer_version', 3, 'string', 'optional'],
  ['domain', 4, 'string', 'optional'],
  ['model_version', 5, 'int64', 'optional'],
  ['doc_string', 6, 'string', 'optional'],
  ['graph', 7, 'GraphProto', 'optional'],
  ['opset_import', 8, 'OperatorSetIdProto', 'repeated'],
  ['metadata_props', 14, 'StringStringEntryProto', 'repeated'],
  ['training_info', 20, 'TrainingInfoProto', 'repeated'],
  ['functions', 25, 'FunctionProto', 'repeated'],
]);

const nodeType = root.lookupType('NodeProto');

function uniqueName(used: Set<string>, base: string): string {
  let candidate = base;
  let suffix = 1;
  while (used.has(candidate)) {
    candidate = `${base}_${suffix++}`;
  }
  used.add(candidate);
  return candidate;
}

/**
 * Rewrite ONNX PRelu nodes into supported primitive operators. ONNX Runtime
 * WebGPU currently lacks a PRelu kernel, while Neg/Relu/Mul/Sub are supported.
 * The identity is exact for every slope value: PRelu(x, a) = Relu(x) -
 * a * Relu(-x).
 */
export function rewritePReluForWebGPU(modelData: ArrayBuffer): {
  data: ArrayBuffer;
  rewrittenNodes: number;
} {
  let model: any;
  try {
    model = modelType.decode(new Uint8Array(modelData));
  } catch {
    return { data: modelData, rewrittenNodes: 0 };
  }

  const graph = model.graph;
  if (!graph || !Array.isArray(graph.node)) {
    return { data: modelData, rewrittenNodes: 0 };
  }

  const usedNames = new Set<string>();
  for (const node of graph.node) {
    if (node.name) usedNames.add(node.name);
    for (const output of node.output || []) usedNames.add(output);
  }

  const rewrittenNodes: any[] = [];
  let rewrittenCount = 0;

  for (let index = 0; index < graph.node.length; index++) {
    const node = graph.node[index];
    if (node.op_type !== 'PRelu' || !node.input || node.input.length < 2 || !node.output?.length) {
      rewrittenNodes.push(node);
      continue;
    }

    const base = uniqueName(usedNames, node.name || `PRelu_${index}`);
    const negative = uniqueName(usedNames, `${base}_negative`);
    const negativeRelu = uniqueName(usedNames, `${base}_negative_relu`);
    const scaledNegative = uniqueName(usedNames, `${base}_scaled_negative`);
    const positiveRelu = uniqueName(usedNames, `${base}_positive_relu`);
    const output = node.output[0];

    const makeNode = (opType: string, input: string[], outputName: string, suffix: string) =>
      nodeType.create({
        input,
        output: [outputName],
        name: uniqueName(usedNames, `${base}_${suffix}`),
        op_type: opType,
      });

    rewrittenNodes.push(
      makeNode('Neg', [node.input[0]], negative, 'neg'),
      makeNode('Relu', [negative], negativeRelu, 'neg_relu'),
      makeNode('Mul', [node.input[1], negativeRelu], scaledNegative, 'mul'),
      makeNode('Relu', [node.input[0]], positiveRelu, 'pos_relu'),
      makeNode('Sub', [positiveRelu, scaledNegative], output, 'sub')
    );
    rewrittenCount++;
  }

  if (rewrittenCount === 0) {
    return { data: modelData, rewrittenNodes: 0 };
  }

  graph.node = rewrittenNodes;
  const encoded = modelType.encode(model).finish();
  const data = encoded.buffer.slice(
    encoded.byteOffset,
    encoded.byteOffset + encoded.byteLength
  ) as ArrayBuffer;
  return { data, rewrittenNodes: rewrittenCount };
}
