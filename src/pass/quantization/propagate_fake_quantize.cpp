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
static const auto initial_quant_params = mv::QuantizationParams{{0}, {1}, {-inf}, {inf}};

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
mv::QuantizationParams extractQuantParams(mv::Data::OpListIterator fqOp, bool merge_in_one) {
    assert(fqOp->getOpType() == "FakeQuantize");

    auto inputs = fqOp->getInputTensor();
    auto attrs = fqOp->getAttrs();

    auto levels = fqOp->get<unsigned>("levels");

    auto input_min = fqOp->getInputTensor(1)->getDoubleData();
    auto input_max = fqOp->getInputTensor(2)->getDoubleData();
    auto output_min = fqOp->getInputTensor(3)->getDoubleData();
    auto output_max = fqOp->getInputTensor(4)->getDoubleData();

    assert(input_min.size() != 0);

    std::vector<int64_t> zero_points;
    std::vector<double> scales;
    std::vector<double> min;
    std::vector<double> max;
    if (merge_in_one) {
        float output_min_value = *std::min_element(output_min.begin(), output_min.end());
        float output_max_value = *std::max_element(output_max.begin(), output_max.end());

        zero_points.push_back(calculateZeroPoint(output_min_value, output_max_value, levels, getDType(Precision::U8)));
        scales.push_back(calculateScales(output_min_value, output_max_value, levels));
        min.push_back(output_min_value);
        max.push_back(output_max_value);
    } else {
        for (size_t i = 0; i < output_min.size(); ++i) {
            float min_value = output_min[i];
            float max_value = output_max[i];

            zero_points.push_back(calculateZeroPoint(min_value, max_value, levels, getDType(Precision::U8)));
            scales.push_back(calculateScales(min_value, max_value, levels));
            min.push_back(min_value);
            max.push_back(max_value);
        }
    }

    return mv::QuantizationParams{zero_points, scales, min, max};
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
    //NOTE: we have the case were is assumption is wrong. See PriorBox
    assert(op->getOutputTensor().size() == 1);

    mv::DataModel dm(model);

    auto current_op = findSinkLayers(dm, op->getOutputTensor(0)).at(0);
    while(current_op->getOpType() != "FakeQuantize" && current_op->getOpType() != "Output")
    {
        current_op = findSinkLayers(dm, current_op->getOutputTensor(0)).at(0);
        assert(current_op->getOutputTensor().size() < 2); // < 2 handles output op. Output op outTensor.size() equals 0
    }

    if (current_op->getOpType() == "FakeQuantize") {
        assert(op->getOpType() != "ConstantInt");
        return extractQuantParams(current_op, op->getOpType() != "Constant");
    }

    return initial_quant_params;
}

bool isEqual(const mv::QuantizationParams& left, const mv::QuantizationParams& right) {
    return left.getScale() == right.getScale()
        && left.getZeroPoint() == right.getZeroPoint()
        && left.getMin() == right.getMin()
        && left.getMax() == right.getMax();
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
        if (op->getOpType() == "Eltwise" && op->getOpType() == "Concat") {
            assert(areAllInputQuantParamsEqual(om, op));
        }

        if ((isQuantizableOp(op) && isOpQuantized(om, op)) || op->getOpType() == "Constant") { // NOTE: float16 case is not handled here
            quant_params = findOutputQuantParams(model, op);

            if (op->getOpType() == "AveragePool" && isEqual(quant_params, initial_quant_params)) {
                quant_params = getParentQuantParams(om, op);
            }

            setQuantizationParams(op, quant_params);
        } else if (op->getOpType() != "Input" && op->getOpType() != "ConstantInt") {
            auto parent = om.getSourceOp(op->getInputTensor(0));
            if (parent->getOpType() == "Input" && op->getOpType() == "Scale")
                continue;

            quant_params = getParentQuantParams(om, op);
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
mv::Data::OpListIterator quantizeConstOp(mv::OpModel& om, mv::Data::OpListIterator operation, mv::QuantizationParams quant_params, mv::DType precision) {
    // TODO: fp16 tensors. need to handle
    assert(operation->getOpType() == "Constant");

    auto originalTensor = operation->getOutputTensor(0);
    assert(originalTensor->getDType() != getDType(Precision::FP16));

    auto shape = operation->getOutputTensor(0)->getShape();

    size_t OC, IC, KH, KW;
    getWeightsDims(shape, OC, IC, KW, KH);

    auto src_data = operation->getOutputTensor(0)->getDoubleData();
    std::vector<int64_t> quantized_weights(src_data.size());
    auto min = quant_params.getMin();
    auto max = quant_params.getMax();

    // Workaround for depthwise convolution
    // Because we expect shpae of depthwise conv to be in {KH, KW, N, 1} format
    // and all other weights have shape like {KW, KW, IC, N} we have OC=1 and IC=N
    // But further logic requires OC be equal to N and IC equal to 1
    if (operation->hasAttr("is_depthwise_weights")) {
        std::swap(OC, IC);
    }

    //TODO: rewrite. We can do it only for ochannel loop
    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t ic = 0; ic < IC; ++ic) {
            for (size_t kh = 0; kh < KH; ++kh) {
                for (size_t kw = 0; kw < KW; ++kw) {
                    const size_t idx = (oc * IC * KW * KH) +
                                       (ic * KH * KW) +
                                       (kh * KW) +
                                        kw;

                    auto scale = quant_params.getScale(oc);
                    auto low = min[min.size() > 1 ? oc : 0];
                    auto high = max[max.size() > 1 ? oc : 0];

                    auto new_value = clamp<double>(src_data[idx], low, high);
                    new_value = std::round((new_value - low) / scale);

                    if (precision == getDType(Precision::U8)) {
                        uint8_t value = clamp<double>(new_value, 0, 255);
                        quantized_weights[idx] = value;
                    } else if (precision == getDType(Precision::I8)) {
                        int8_t value = clamp<double>(new_value, -128, 127);
                        quantized_weights[idx] = value;
                    } else {
                        throw std::runtime_error("Unexpected dtype");
                    }
                }
            }
        }
    }

    auto quantized_const_tensor = om.constantInt(quantized_weights, shape, precision, originalTensor->getOrder(), quant_params, operation->getName()+":quantized");
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

        auto quant_params = extractQuantParams(fq, false);
        quantizeConstOp(om, op, quant_params, getDType(Precision::U8));
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
    // Do nothing with input. Keep parameters

    assert(om.getOps("Output").size() == 1);
    auto output = om.getOps("Output").at(0);
    setQuantizationParams(output, {{}, {}, {}, {}});
}

void removeFQ(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {
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

    auto input = om.getInput();
    auto current_op = findSinkLayers(dm, input->getOutputTensor(0)).at(0);
    if (current_op->getOpType() != "Scale" || !current_op->getInputTensor(1)->isDoubleType()) {
        return;
    }

    // Quantize input scale
    auto scaleTensor = current_op->getInputTensor(1);
    auto originalConstOp = om.getSourceOp(scaleTensor);
    auto scaleData = scaleTensor->getDoubleData();
    auto quantizedScaleData = std::vector<int64_t>(scaleData.size(), 1);

    mv::QuantizationParams scalesQuantParams = {{0}, scaleData, {-inf}, {inf}};
    auto quantized_const_tensor = om.constantInt(quantizedScaleData, scaleTensor->getShape(), getDType(Precision::U8), scaleTensor->getOrder(), scalesQuantParams, originalConstOp->getName()+":quantized");
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
        auto bias_op = quantizeBias(model, current_op, initial_quant_params, scalesQuantParams);
        setQuantizationParams(bias_op, getParentQuantParams(om, bias_op));
    }
}

void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&) {
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
