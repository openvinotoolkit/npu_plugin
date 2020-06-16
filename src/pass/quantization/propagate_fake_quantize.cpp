#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <math.h>

static void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(FakeQuantize)
        .setFunc(quantizeGraphFcn)
        .setDescription(
            "This pass propagate parameters from FakeQuantize ops to all other operations and quantize const data"
        );
    }
}

static double inf = std::numeric_limits<double>::infinity();

// This is needed to avoid a static initializer fiasco.
static const mv::QuantizationParams& initial_quant_params() {
    static mv::QuantizationParams init{{0}, {1}, {-inf}, {inf}};
    return init;
}

enum class Precision {
    Default,
    U8,
    I8,
    FP16,
    I32,
    FP32,
};

mv::DType getDType(Precision p) {
    static const std::map<Precision, mv::DType> types {
            {Precision::Default, mv::DType("Default")},
            {Precision::U8, mv::DType("UInt8")},
            {Precision::I8, mv::DType("Int8")},
            {Precision::FP16, mv::DType("Float16")},
            {Precision::I32, mv::DType("Int32")},
            {Precision::FP32, mv::DType("Float32")},
    };

    return types.at(p);
}

int64_t calculateZeroPoint(float low, float high, int levels, mv::DType dtype) {
    int64_t zeroPoint = 0;

    // Typical condition for symmetric case is low < 0, high > 0
    if (dtype == getDType(Precision::U8)) {
        //  MCM team provide this formula, need check
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * low / (high - low);
            zeroPoint = std::round(x);
        } else if (low >= 0.f) {
            zeroPoint = 0;  // TODO Why not assert?
        } else if (high <= 0.f) {
            zeroPoint = (levels - 1);  // TODO Why not assert?
        }
    }
    if (dtype == getDType(Precision::I8)) {
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * ((high + low) * 0.5f) / (high - low);
            zeroPoint = std::round(x);
        } else if (low > 0.f) {
            zeroPoint = 127 - (levels - 1);  // TODO Why not assert?
        } else if (high < 0.f) {
            zeroPoint = 127;  // TODO Why not assert?
        }
    }

    return zeroPoint;
}

double calculateScales(float low, float high, int levels) {
    return static_cast<double>((high - low) / (levels - 1));
}

//NOTE: workaround. merge_in_one is true for activations and false for weights
mv::QuantizationParams extractQuantParams(mv::Data::OpListIterator fqOp, bool merge_in_one, bool extract_input_params = false) {
  assert(fqOp->getOpType() == "FakeQuantize");

  auto inputs = fqOp->getInputTensor();
  auto attrs = fqOp->getAttrs();

  auto levels = fqOp->get<unsigned>("levels");

  std::vector<double> min_range;
  std::vector<double> max_range;

  if (extract_input_params) {
      min_range = fqOp->getInputTensor(1)->getDoubleData();
      max_range = fqOp->getInputTensor(2)->getDoubleData();
  } else {
      min_range = fqOp->getInputTensor(3)->getDoubleData();
      max_range = fqOp->getInputTensor(4)->getDoubleData();
  }

  assert(min_range.size() != 0);

  std::vector<int64_t> zero_points;
  std::vector<double> scales;
  std::vector<double> min;
  std::vector<double> max;
  if (merge_in_one) {
    float output_min_value = *std::min_element(min_range.begin(), min_range.end());
    float output_max_value = *std::max_element(max_range.begin(), max_range.end());

    zero_points.push_back(calculateZeroPoint(output_min_value, output_max_value, levels, getDType(Precision::U8)));
    scales.push_back(calculateScales(output_min_value, output_max_value, levels));
    min.push_back(output_min_value);
    max.push_back(output_max_value);
  } else {
    for (size_t i = 0; i < min_range.size(); ++i) {
      float min_value = min_range[i];
      float max_value = max_range[i];

      zero_points.push_back(calculateZeroPoint(min_value, max_value, levels, getDType(Precision::U8)));
      scales.push_back(calculateScales(min_value, max_value, levels));
      min.push_back(min_value);
      max.push_back(max_value);
    }
  }

  return mv::QuantizationParams{zero_points, scales, min, max};
}

mv::QuantizationParams extractQuantParamsI(mv::Data::OpListIterator fqOp, bool merge_in_one) {
    return extractQuantParams(fqOp, merge_in_one, true);
}

mv::QuantizationParams extractQuantParamsO(mv::Data::OpListIterator fqOp, bool merge_in_one) {
    return extractQuantParams(fqOp, merge_in_one, false);
}

template<typename T>
T clamp(const T& value, const T& min, const T& max) {
    assert(min < max);
    return std::max(min, std::min(max, value));
}

static bool isQuantizableOp(mv::Data::OpListIterator op) {
    static std::set<std::string> quantizable_ops{"Conv", "FullyConnected", "Eltwise" , "AveragePool", "DepthwiseConv", "Scale"};
    return quantizable_ops.count(op->getOpType());
}

bool isOpQuantized(mv::OpModel& om, mv::Data::OpListIterator op) {
    if (!isQuantizableOp(op)) {
        return false;
    }

    if (op->getOpType() == "AveragePool") {
        return om.getSourceOp(op->getOutputTensor(0))->getOpType() == "FakeQuantize";
    }

    assert(op->getInputTensor().size() > 1);
    return (om.getSourceOp(op->getInputTensor(1))->getOpType() == "FakeQuantize") ||
            op->getInputTensor(1)->getDType() == getDType(Precision::U8) ||
            op->getInputTensor(1)->getDType() == getDType(Precision::I8);
}

mv::QuantizationParams findOutputQuantParams(mv::ComputationModel& model, mv::Data::OpListIterator op) {
    //NOTE: If we have more than one child we have several cases
    //ALL branches without FQ -> initial quant params and float output
    //FQ only on one branch -> set this quant_params
    //FQ on all branches with same quant params ->set this quant params
    //FQ with different quant params -> assert

    mv::DataModel dm(model);
    auto current_ops = findSinkLayers(dm, op->getOutputTensor(0));

    while(current_ops.size() == 1 && current_ops[0]->getOpType() != "FakeQuantize" && current_ops[0]->getOpType() != "Output") {
        assert(!isQuantizableOp(current_ops[0]));
        current_ops = findSinkLayers(dm, current_ops[0]->getOutputTensor(0));
        assert(current_ops[0]->getOutputTensor().size() < 2);
    }

    std::vector<mv::QuantizationParams> outQuantParams;
    for (size_t i = 0; i < current_ops.size(); i++) {
        if (current_ops[i]->getOpType() == "FakeQuantize") {
            outQuantParams.push_back(extractQuantParamsO(current_ops[0], op->getOpType() != "Constant"));
        }
    }

    // NO FQ on branches
    if (outQuantParams.empty()) {
        return initial_quant_params();
    }

    for (size_t i = 1; i < outQuantParams.size(); i++) {
        if (!isEqual(outQuantParams[0], outQuantParams[i])) {
            throw std::runtime_error("Different quant params on branches");
        }
    }
    return outQuantParams[0];
}

mv::QuantizationParams getParentQuantParams(mv::OpModel& om, const mv::Data::OpListIterator& op, size_t parent_idx = 0) {
    assert(op->getInputTensor().size() > 0);
    auto parent = om.getSourceOp(op->getInputTensor(parent_idx));
    assert(parent->hasAttr("quantParams"));
    return parent->get<mv::QuantizationParams>("quantParams");
}

void setQuantizationParams(mv::Data::OpListIterator& op, mv::QuantizationParams quant_params) {
    op->set<mv::QuantizationParams>("quantParams", quant_params);
    auto out_tensors = op->getOutputTensor();
    for (auto& tensor : out_tensors) {
        tensor->set<mv::QuantizationParams>("quantParams", quant_params);
    }
}

bool areAllInputQuantParamsEqual(mv::OpModel om, mv::Data::OpListIterator op) {
    std::vector<mv::QuantizationParams> input_params;
    for (size_t i = 0; i < op->getInputTensor().size(); ++i) {
        input_params.push_back(getParentQuantParams(om, op, i));
    }

    return std::adjacent_find(input_params.begin(), input_params.end(),
            [](const mv::QuantizationParams& left, const mv::QuantizationParams& right) {
        return !isEqual(left, right);
    }) != input_params.end();
}

//NOTE: here is FQ operation parameters propagation algorithm.
// Since FQ was design to be propagted parameters to a next layer to quantize it (FQ -> Layer(that should be quantized))
// this approach aims to provide suitable output quantization parameters for all operations in terms of mcm.
// The basic idea is to find output quantization parameters for each operation in the graph
// (If this operation is "quantizable". It means that it can  greatly affect output range and be executed in integer)
// (In this terms all quantization agnostic operations such as MaxPool are not qunatizable)
// For each operation we are trying our heruristic to set quantization operation
// If operation is quantizable and needs to be quantized we are trying to find the next FQ quantize operation
// and extract quantization parameters for current operation.
// Else we just get quntization parameters from the parent
// Example: ->Conv->Bias->Relu->FQ
// Conv will get parameters from FQ. Bias from Conv. And Relu from Bias.
// Exceptions:
// For AveragePool. If we didn't find output quant params, just get parents qunat_params
// For scale and bias right after the Input op we save parameters propagated from the Kmb-Plugin as a workaround
void propagateParameters(mv::ComputationModel& model) {
    mv::OpModel om(model);
    mv::QuantizationParams quant_params{{}, {}, {}, {}};
    auto sorted_ops = om.topologicalSort();
    for (auto& op : sorted_ops) {
        if (op->getOpType() == "Eltwise" || op->getOpType() == "Concat") {
            assert(areAllInputQuantParamsEqual(om, op));
        }

        if ((isQuantizableOp(op) && isOpQuantized(om, op)) || op->getOpType() == "Constant" // NOTE: float16 case is not handled here
            || op->getOpType() == "Interp") { //Interp might be used for re-quantize, need the quant params
            quant_params = findOutputQuantParams(model, op);

            if (op->getOpType() == "AveragePool" && isEqual(quant_params, initial_quant_params())) {
                quant_params = getParentQuantParams(om, op);
            }

#if 1
            if (op->getName() == "conv4_3"){
                quant_params = mv::QuantizationParams({{0},{2.13719},{},{}});
                std::cout << op->getName() << " (" << op->getOpType() << "):  (" << quant_params.getZeroPoint()[0] << ", " << quant_params.getScale()[0] << ")" << std::endl;
            }
#endif
            setQuantizationParams(op, quant_params);
        } else if (op->getOpType() != "Input" && op->getOpType() != "ConstantInt") {
            auto parent = om.getSourceOp(op->getInputTensor(0));
            if (parent->getOpType() == "Input" && op->getOpType() == "Scale")
                continue;

            quant_params = getParentQuantParams(om, op);
#if 1
            if (op->getName() == "conv4_3_norm"){
                quant_params = mv::QuantizationParams({{0},{0.026569},{},{}});
                std::cout << op->getName() << " (" << op->getOpType() << "):  (" << quant_params.getZeroPoint()[0] << ", " << quant_params.getScale()[0] << ")" << std::endl;
            }
#endif
            setQuantizationParams(op, quant_params);
        }
    }
}

void getWeightsDims(const mv::Shape& shape, size_t& OC, size_t& IC, size_t& KH, size_t& KW) {
    auto new_shape = shape;
    if (new_shape.ndims() == 2) {
        new_shape = new_shape.augment(new_shape, 4);
    }

    OC = new_shape[mv::KERNEL_OUTPUT_CHANNELS];
    IC = new_shape[mv::KERNEL_INPUT_CHANNELS];
    KH = new_shape[mv::KERNEL_HEIGHT];
    KW = new_shape[mv::KERNEL_WIDTH];
}

//NOTE: Bias is not a const op.
mv::Data::OpListIterator quantizeConstOp(mv::OpModel& om, mv::Data::OpListIterator operation,
        mv::QuantizationParams quant_paramsI, mv::QuantizationParams quant_paramsO, mv::DType precision) {
    // TODO: fp16 tensors. need to handle
    assert(operation->getOpType() == "Constant");

    auto originalTensor = operation->getOutputTensor(0);
    assert(originalTensor->getDType() != getDType(Precision::FP16));

    auto shape = operation->getOutputTensor(0)->getShape();

    size_t OC, IC, KH, KW;
    getWeightsDims(shape, OC, IC, KW, KH);

    auto src_data = operation->getOutputTensor(0)->getDoubleData();
    std::vector<int64_t> quantized_weights(src_data.size());
    auto min = quant_paramsI.getMin();
    auto max = quant_paramsI.getMax();

    // Workaround for depthwise convolution
    // Because we expect shpae of depthwise conv to be in {KH, KW, N, 1} format
    // and all other weights have shape like {KW, KW, IC, N} we have OC=1 and IC=N
    // But further logic requires OC be equal to N and IC equal to 1
    if (operation->hasAttr("is_depthwise_weights")) {
        std::swap(OC, IC);
    }

    //TODO: rewrite. We can do it only for ochannel loop
    for (size_t oc = 0; oc < OC; ++oc) {
        auto scale = quant_paramsI.getScale(oc);
        auto low = min[min.size() > 1 ? oc : 0];
        auto high = max[max.size() > 1 ? oc : 0];

        for (size_t i = oc * IC * KW * KH; i < (oc+1) * IC * KW * KH; ++i) {
            auto new_value = clamp<double>(src_data[i], low, high);
            new_value = std::round((new_value - low) / scale);

            if (precision == getDType(Precision::U8)) {
                uint8_t value = clamp<double>(new_value, 0, 255);
                quantized_weights[i] = value;
            } else if (precision == getDType(Precision::I8)) {
                int8_t value = clamp<double>(new_value, -128, 127);
                quantized_weights[i] = value;
            } else {
                throw std::runtime_error("Unexpected dtype");
            }
        }
    }

    auto quantized_const_tensor = om.constantInt(quantized_weights, shape, precision, originalTensor->getOrder(), quant_paramsO, operation->getName()+":quantized");
    if(operation->hasAttr("opId")) {
        unsigned currentOpId = operation->get<unsigned>("opId");
        quantized_const_tensor->set<unsigned>("opId", currentOpId);
        om.getSourceOp(quantized_const_tensor)->set<unsigned>("opId", currentOpId);
    }

    return mv::linkNewOperationsReplacement(mv::Data::OpListIterator(), quantized_const_tensor, om, operation);
}

void quantizeConst(mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto fq_ops = om.getOps("FakeQuantize");

    for (auto& fq : fq_ops) {
        auto op = om.getSourceOp(fq->getInputTensor(0));
        if (op->getOpType() != "Constant") {
            continue;
        }

        auto data_tensor = op->getOutputTensor(0);
        if (data_tensor->getDType() == getDType(Precision::I8) ||
            data_tensor->getDType() == getDType(Precision::U8)) {
            // Weights would already be quantized
            // Do nothing
            return;
        }

        auto quant_paramsI = extractQuantParamsI(fq, false);
        auto quant_paramsO = extractQuantParamsO(fq, false);
        quantizeConstOp(om, op, quant_paramsI, quant_paramsO, getDType(Precision::U8));
    }
}

//NOTE: To quantize bias we use quantization parameters of input op and activations weights;
// Input-> Activation(e.g. Conv) -> Bias
// In current mcm approach it doesn't matter which qunat_params were set on bias
mv::Data::OpListIterator quantizeBias(mv::ComputationModel& model, mv::Data::OpListIterator biasOp, mv::QuantizationParams input_quant_params,
                                      mv::QuantizationParams weights_params) {
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto tensor_data = biasOp->getInputTensor(1)->getData();

    bool is_broadcasted = input_quant_params.isScalePerTensor() && weights_params.isScalePerTensor();

    if (!input_quant_params.isScalePerTensor() && input_quant_params.getScale().size() != tensor_data.size()) {
        throw std::runtime_error("Bias and Activation quant params size mismatch");
    }

    if (!weights_params.isScalePerTensor() && weights_params.getScale().size() != tensor_data.size()) {
        throw std::runtime_error("Bias and Weights quant params size mismatch");
    }

    std::vector<int64_t> newBiasData(tensor_data.size(), 0);
    std::vector<double> biasScales;
    std::vector<int64_t> zeroPoints;
    auto bias_dtype = biasOp->getInputTensor(1)->getDType();

    if (bias_dtype == getDType(Precision::FP32)) {
        auto bias_data = biasOp->getInputTensor(1)->getDoubleData();

        for (size_t i = 0; i < bias_data.size(); ++i) {
            auto activation_scale = input_quant_params.getScale(i);
            auto weights_scale = weights_params.getScale(i);

            auto bias_scale = activation_scale * weights_scale;
            biasScales.push_back(bias_scale);
            zeroPoints.push_back(0);
            newBiasData[i] = std::round(bias_data[i] / bias_scale);
        }

        if (is_broadcasted) {
            biasScales.resize(1);
            zeroPoints.resize(1);
        }

        auto original_tensor = biasOp->getInputTensor(1);

        auto quantized_data = om.constantInt(newBiasData,
                                                original_tensor->getShape(),
                                                getDType(Precision::I32),
                                                original_tensor->getOrder(),
                                                input_quant_params,
                                                om.getSourceOp(biasOp->getInputTensor(1))->getName()+":quantized");

        auto quantize_bias_tensor = om.bias(biasOp->getInputTensor(0),
                                               quantized_data,
                                               quantized_data->getDType(),
                                               input_quant_params,
                                               biasOp->getName()+":quantized");

        auto parent_op = mv::linkNewOperationsReplacement(om.getSourceOp(biasOp->getInputTensor(0)), quantize_bias_tensor, om, biasOp);
        return findSinkLayers(dm, parent_op->getOutputTensor(0)).at(0);
    } else if (bias_dtype == getDType(Precision::I32)) {
        // Do nothing
    } else {
        std::runtime_error("Unsupported bias data type");
    }

    return biasOp;
}

void quantizeBias(mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto biasOps = om.getOps("Bias");

    for (auto& biasOp : biasOps) {
        auto parent = om.getSourceOp(biasOp->getInputTensor(0));

        if (isOpQuantized(om, parent)) {
            auto activationOp = om.getSourceOp(biasOp->getInputTensor(0));
            auto input_quant_params = getParentQuantParams(om, activationOp);

            auto weights_op = om.getSourceOp(activationOp->getInputTensor(1));
            auto weights_params = weights_op->get<mv::QuantizationParams>("quantParams");

            quantizeBias(model, biasOp, input_quant_params, weights_params);
        }
    }
}

void quantizeIO(mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto inputs = om.getOps("Input");
    mv::DataModel dm(om);
    for (size_t idx = 0; idx < inputs.size(); idx++) {
        auto input = inputs.at(idx);
        auto current_ops = findSinkLayers(dm, input->getOutputTensor(0));

        mv::QuantizationParams inputQuantParams = initial_quant_params();
        if(current_ops.size() == 1 && current_ops[0]->getOpType() == "FakeQuantize") {
            inputQuantParams = extractQuantParams(current_ops[0], input->getOpType() != "Constant");
        }

        //Fuse scaleshift(multiply+add) into quantization parameters to boost performance 
        if(current_ops.size() == 1 && current_ops[0]->getOpType() == "Scale") {
            auto child_ops = findSinkLayers(dm, current_ops[0]->getOutputTensor(0));
            if(child_ops.size() == 1 && child_ops[0]->getOpType() == "Bias") {
                auto next_child_ops = findSinkLayers(dm, child_ops[0]->getOutputTensor(0));
                if(next_child_ops.size() == 1 && next_child_ops[0]->getOpType() == "FakeQuantize") {
                    auto inputC = input->getOutputTensor(0)->getShape()[2];

                    if (current_ops[0]->getInputTensor(1)->isDoubleType()) {
                        std::runtime_error("Unsupported fuse scaleshift DType");
                    }
                    std::vector<int64_t> scaleData = current_ops[0]->getInputTensor(1)->getIntData();
                    auto biasData = child_ops[0]->getInputTensor(1)->getIntData();
                    std::vector<double> realScaleValue(inputC);
                    std::vector<double> realBiasValue(inputC);

                    //calculate real scale values
                    auto scaleDataQuantParams = om.getSourceOp(current_ops[0]->getInputTensor(1))->get<mv::QuantizationParams>("quantParams");
                    auto s = scaleDataQuantParams.get<std::vector<double>>("scale");
                    auto zp = scaleDataQuantParams.get<std::vector<int64_t>>("zeroPoint");
                    std::vector<double> s_extend(inputC);
                    std::vector<int64_t> zp_extend(inputC);
                    std::vector<int64_t> scaleData_extend(inputC);

                    if (s.size() < inputC) {
                        assert(s.size() == 1);
                        for (size_t i = 0; i < inputC; i++) {
                            s_extend[i] = s[0];
                        }
                    }
                    else {
                        s_extend = s;
                    }

                    if (zp.size() < inputC) {
                        assert(zp.size() == 1);
                        for (size_t i = 0; i < inputC; i++) {
                            zp_extend[i] = zp[0];
                        }
                    }
                    else {
                        zp_extend = zp;
                    }

                    if (scaleData.size() < inputC) {
                        assert(scaleData.size() == 1);
                        for (size_t i = 0; i < inputC; i++) {
                            scaleData_extend[i] = scaleData[0];
                        }
                    }
                    else {
                        scaleData_extend = scaleData;
                    }

                    for (size_t i = 0; i < inputC; i++) {
                        realScaleValue[i] = (scaleData_extend[i] - zp_extend[i]) * s_extend[i];
                    }

                    //calculate real bias values
                    auto biasDataQuantParams = om.getSourceOp(child_ops[0]->getInputTensor(1))->get<mv::QuantizationParams>("quantParams");
                    s = biasDataQuantParams.get<std::vector<double>>("scale");
                    zp = biasDataQuantParams.get<std::vector<int64_t>>("zeroPoint");
                    std::vector<int64_t> biasData_extend(inputC);

                    if (s.size() < inputC) {
                        assert(s.size() == 1);
                        for (size_t i = 0; i < inputC; i++) {
                            s_extend[i] = s[0];
                        }
                    }
                    else {
                        s_extend = s;
                    }

                    if (zp.size() < inputC) {
                        assert(zp.size() == 1);
                        for (size_t i = 0; i < inputC; i++) {
                            zp_extend[i] = zp[0];
                        }
                    }
                    else {
                        zp_extend = zp;
                    }

                    if (biasData.size() < inputC) {
                        assert(biasData.size() == 1);
                        for (size_t i = 0; i < inputC; i++) {
                            biasData_extend[i] = biasData[0];
                        }
                    }
                    else {
                        biasData_extend = biasData;
                    };

                    for (size_t i = 0; i < inputC; i++) {
                        realBiasValue[i] = (biasData_extend[i] - zp_extend[i]) * s_extend[i];
                    }

                    //update new zp/scale/min/max
                    auto levels = next_child_ops[0]->get<unsigned>("levels");
                    std::vector<int64_t> zero_points;
                    std::vector<double> scales;
                    std::vector<double> min;
                    std::vector<double> max;

                    float output_min_value = (realBiasValue[0] * realScaleValue[0] + 
                    realBiasValue[1] * realScaleValue[1] + realBiasValue[2] * realScaleValue[2]) / 3;

                    float output_max_value = ((realBiasValue[0] + 255) * realScaleValue[0] + 
                    (realBiasValue[1] + 255) * realScaleValue[1] + (realBiasValue[2] + 255) * realScaleValue[2]) / 3;

                    zero_points.push_back(calculateZeroPoint(output_min_value, output_max_value, levels, getDType(Precision::U8)));
                    scales.push_back(calculateScales(output_min_value, output_max_value, levels));
                    min.push_back(output_min_value);
                    max.push_back(output_max_value);

                    inputQuantParams = mv::QuantizationParams{zero_points, scales, min, max};

                    linkNewOperationsRemove(input, input->getOutputTensor(0), om, current_ops[0]);
                    linkNewOperationsRemove(input, input->getOutputTensor(0), om, child_ops[0]);
                }
            }
        }

        setQuantizationParams(input, inputQuantParams);
    }

    assert(om.getOps("Output").size() == 1);
    auto output = om.getOps("Output").at(0);
    setQuantizationParams(output, {{}, {}, {}, {}});
}

void removeFQ(const mv::pass::PassEntry&, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto fq_ops = om.getOps("FakeQuantize");

    for (auto& fq : fq_ops) {
        auto parent = om.getSourceOp(fq->getInputTensor(0));
        linkNewOperationsRemove(parent, parent->getOutputTensor(0), om, fq);
    }
}

void quantizeInputScaleShift(mv::ComputationModel& model) {
    mv::OpModel om(model);
    mv::DataModel dm(model);

    // Quantize all Scale
    // Extend Scale to whole netwrok, not only for Input, networks e.g.Facenet need it 
    auto scaleOps = om.getOps("Scale");
    for (auto& current_op : scaleOps) {
        if (current_op->getInputTensor(1)->isDoubleType()) {
            auto scaleTensor = current_op->getInputTensor(1);
            auto originalConstOp = om.getSourceOp(scaleTensor);
            auto scaleData = scaleTensor->getDoubleData();
            auto quantizedScaleData = std::vector<int64_t>(scaleData.size(), 1);

            mv::QuantizationParams scalesQuantParams = {{0}, scaleData, {-inf}, {inf}};
            auto quantized_const_tensor =
                om.constantInt(quantizedScaleData, scaleTensor->getShape(), getDType(Precision::U8),
                    scaleTensor->getOrder(), scalesQuantParams, originalConstOp->getName() + ":quantized");
            if (originalConstOp->hasAttr("opId")) {
                unsigned currentOpId = originalConstOp->get<unsigned>("opId");
                quantized_const_tensor->set<unsigned>("opId", currentOpId);
                om.getSourceOp(quantized_const_tensor)->set<unsigned>("opId", currentOpId);
            }
            mv::linkNewOperationsReplacement(mv::Data::OpListIterator(), quantized_const_tensor, om, originalConstOp);

            setQuantizationParams(current_op, findOutputQuantParams(om, current_op));

            // Quantize input bias
            current_op = findSinkLayers(dm, current_op->getOutputTensor(0)).at(0);
            if (current_op->getOpType() == "Bias" && current_op->getInputTensor(1)->isDoubleType()) {
                auto bias_op = quantizeBias(model, current_op, initial_quant_params(), scalesQuantParams);
                setQuantizationParams(bias_op, getParentQuantParams(om, bias_op));
            }
        }
    }
}

void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&) {
    mv::OpModel om(model);
    auto fq_ops = om.getOps("FakeQuantize");
    if (fq_ops.empty())
        return;

    quantizeInputScaleShift(model);
    quantizeIO(model);

    propagateParameters( model);

    quantizeConst(model);
    quantizeBias(model);

    removeFQ(pass, model);
}
