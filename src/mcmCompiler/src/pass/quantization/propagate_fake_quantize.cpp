#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/pass_quantization.hpp"

static void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);
bool isOpPassthrough(const mv::Data::OpListIterator& op);

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

template<typename T>
T clamp(const T& value, const T& min, const T& max) {
    if (min > max)
        throw std::runtime_error("Unsupported clamp params");
    return std::max(min, std::min(max, value));
}

static bool isQuantizableOp(mv::Data::OpListIterator op) {
    static std::set<std::string> quantizable_ops{"Conv", "FullyConnected", "Eltwise" , "AveragePool", "DepthwiseConv", "Scale", "Deconv"};
    return quantizable_ops.count(op->getOpType());
}

bool isOpInputQuantized(mv::OpModel& om, const mv::Data::OpListIterator& op) {
    if (!isQuantizableOp(op)) {
        return false;
    }

    // Navigate through chain of FQ invariant parents
    auto sourceOp = om.getSourceOp(op->getInputTensor(0));
    while (isOpPassthrough(sourceOp) && sourceOp->inputSlots())
        sourceOp = om.getSourceOp(sourceOp->getInputTensor(0));

    return (sourceOp->getOpType() == "FakeQuantize") ||
            op->getInputTensor(0)->getDType() == getDType(mv::Precision::U8) ||
            op->getInputTensor(0)->getDType() == getDType(mv::Precision::I8);
}

bool isOpPassthrough(const mv::Data::OpListIterator& op)
{
    std::vector<std::string> passthroughOps = {
        "Bias", "Relu", "LeakyRelu", "Concat", "Maximum", "Minimum", "ReorgYolo", "Reshape", "Permute", "Interp", "Resample", "MaxPool", "Mish", "Sigmoid"
    };

    return std::find(passthroughOps.begin(), passthroughOps.end(), op->getOpType()) != passthroughOps.end() ||
           op->isImplicit();
}

std::vector<mv::Data::OpListIterator> findOutputFakeQuantize(mv::DataModel& dm, const mv::Data::OpListIterator& op) {
    std::vector<mv::Data::OpListIterator> fqOps;

    auto childOps = mv::findSinkLayers(dm, op->getOutputTensor(0));
    for (auto& childOp : childOps) {
        if (childOp->getOpType() == "Output" || childOp->getOpType() == "ImplicitOutput")
            continue;
        else if (childOp->getOpType() == "FakeQuantize") {
            fqOps.push_back(childOp);
        } else if (isOpPassthrough(childOp)) {
            auto childfqOps = findOutputFakeQuantize(dm, childOp);
            fqOps.insert(fqOps.end(), childfqOps.begin(), childfqOps.end());
        }
    }

    return fqOps;
}

// FuseScaleShift pass will set input range to [0; levels-1] after fusing ScaleShift into the next Convolution
bool hasScaleShiftFused(const mv::Data::OpListIterator& fqOp) {
    const auto inputMin = fqOp->getInputTensor(1)->getDoubleData();
    const auto inputMax = fqOp->getInputTensor(2)->getDoubleData();
    const auto levels = fqOp->get<unsigned>("levels");
    return inputMin.size() == 1 && inputMin[0] == 0 &&
           inputMax.size() == 1 && inputMax[0] == levels-1;
}

void propagateToParentOps(mv::OpModel& om, mv::DataModel& dm, const mv::Data::OpListIterator& opIt, const mv::DType& dType, const mv::QuantizationParams& quantParams)
{
    std::vector<mv::Data::TensorIterator> inputTensors = {opIt->getInputTensor(0)};
    if (opIt->getOpType() == "Eltwise" || opIt->getOpType() == "Concat")
        inputTensors = opIt->getInputTensor();

    for (auto inputTensor : inputTensors) {
        inputTensor->setDType(dType);
        if (!(opIt->getOpType() == "Relu" && om.getSourceOp(inputTensor)->getOpType() == "Bias" && om.getSourceOp(inputTensor).childrenSize() > 1))
            inputTensor->setQuantParams(quantParams);
        else
            continue;

        const auto parentOp = om.getSourceOp(inputTensor);
        if (parentOp->getOpType() == "Input" || parentOp->getOpType() == "ImplicitInput")
            continue;

        if (parentOp->outputSlots() > 1)
            continue;

        if (isOpPassthrough(parentOp))
            propagateToParentOps(om, dm, parentOp, dType, quantParams);
    }
}

void propagateToChildOps(mv::DataModel& dm, const mv::Data::OpListIterator& opIt, const mv::DType& dType, const mv::QuantizationParams& quantParams)
{
    if (opIt->outputSlots() > 1)
        return;

    opIt->getOutputTensor(0)->setDType(dType);
    opIt->getOutputTensor(0)->setQuantParams(quantParams);

    auto childOps = mv::findSinkLayers(dm, opIt->getOutputTensor(0));
    for (auto& childOp : childOps) {
        if (childOp->getOpType() == "Output" || childOp->getOpType() == "ImplicitOutput")
            continue;
        if (isOpPassthrough(childOp))
            propagateToChildOps(dm, childOp, dType, quantParams);
    }
}

bool checkChildOps(mv::DataModel& dm, const mv::Data::OpListIterator& opIt)
{
    auto childOps = mv::findSinkLayers(dm, opIt->getOutputTensor(0));
    bool allDepthwiseChild = true;
    for (auto& childOp : childOps) {
        if(isOpPassthrough(childOp))
            allDepthwiseChild &= checkChildOps(dm, childOp);
        else if (childOp->getOpType() != "DepthwiseConv") {
            return false;
        }
    }

    return allDepthwiseChild;
}

void propagateActivationParameters(mv::ComputationModel& model) {
    mv::OpModel om(model);
    mv::DataModel dm(model);

    std::vector<mv::Data::OpListIterator> fqOps;
    const auto sortedOps = om.topologicalSort();
    for (auto& opIt : sortedOps) {
        if (opIt->getOpType() == "FakeQuantize")
            fqOps.push_back(opIt);
    }

    for (auto& fqOp : fqOps) {
        const auto parentOp = om.getSourceOp(fqOp->getInputTensor(0));
        if (parentOp->getOpType() == "Constant" ||
            parentOp->getOpType() == "ConstantInt" ||
            parentOp->getOpType() == "ConstantDataElement")
            continue;

        mv::QuantizationParams inputQuantParams = mv::QuantizationParams::initial();
        mv::QuantizationParams outputQuantParams = mv::QuantizationParams::initial();
        bool allDepthwiseChild = checkChildOps(dm, fqOp);

        if (allDepthwiseChild) {
            inputQuantParams = extractQuantParamsI(fqOp, false);
            outputQuantParams = extractQuantParamsO(fqOp, false);
        } else {
            inputQuantParams = extractQuantParamsI(fqOp, true);
            outputQuantParams = extractQuantParamsO(fqOp, true);
        }

        if (isEqual(inputQuantParams, outputQuantParams)) {
            propagateToParentOps(om, dm, fqOp, getDType(mv::Precision::U8), inputQuantParams);
            propagateToChildOps(dm, fqOp, getDType(mv::Precision::U8), inputQuantParams);
        } else if (hasScaleShiftFused(fqOp)) {
            propagateToParentOps(om, dm, fqOp, getDType(mv::Precision::U8), outputQuantParams);
            propagateToChildOps(dm, fqOp, getDType(mv::Precision::U8), outputQuantParams);
        }
    }
}

void getWeightsDims(const mv::Shape& shape, size_t& OC, size_t& IC, size_t& KH, size_t& KW) {
    auto new_shape = shape;
    if (new_shape.ndims() == 2) {
        new_shape = mv::Shape::augment(new_shape, 4);
    }

    OC = new_shape[mv::KERNEL_OUTPUT_CHANNELS];
    IC = new_shape[mv::KERNEL_INPUT_CHANNELS];
    KH = new_shape[mv::KERNEL_HEIGHT];
    KW = new_shape[mv::KERNEL_WIDTH];
}

//NOTE: Bias is not a const op.
mv::Data::OpListIterator quantizeConstOp(mv::OpModel& om, const mv::Data::OpListIterator& operation,
        const mv::QuantizationParams& quant_paramsI, const mv::QuantizationParams& quant_paramsO, const mv::DType& precision) {
    // TODO: fp16 tensors. need to handle
    if (operation->getOpType() != "Constant" &&
        operation->getOpType() != "ConstantDataElement")
        throw std::runtime_error("quantizeConstOp expected Constant layer type");

    auto originalTensor = operation->getOutputTensor(0);
    if (originalTensor->getDType() == getDType(mv::Precision::FP16))
        throw std::runtime_error("quantizeConstOp expected not floating point type");

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

    for (size_t oc = 0; oc < OC; ++oc) {
        auto scale = quant_paramsI.getScale(oc);
        auto low = min[min.size() > 1 ? oc : 0];
        auto high = max[max.size() > 1 ? oc : 0];

        for (size_t i = 0; i < IC * KW * KH; ++i) {
            auto w = clamp<double>(src_data[oc * IC * KW * KH + i], low, high);
            quantized_weights[oc * IC * KW * KH + i] = std::round((w - low) / scale);
        }
    }

    auto quantized_const_tensor = om.constantInt(operation->getName() + ":quantized", quantized_weights, shape, precision, originalTensor->getOrder());
    quantized_const_tensor->setQuantParams(quant_paramsO);
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
        if (op->getOpType() != "Constant" &&
            op->getOpType() != "ConstantDataElement") {
            continue;
        }

        auto data_tensor = op->getOutputTensor(0);
        if (data_tensor->getDType() == getDType(mv::Precision::I8) ||
            data_tensor->getDType() == getDType(mv::Precision::U8)) {
            // Weights would already be quantized
            // Do nothing
            continue;
        }

        auto quant_paramsI = extractQuantParamsI(fq, false);
        auto quant_paramsO = extractQuantParamsO(fq, false);
        quantizeConstOp(om, op, quant_paramsI, quant_paramsO, getDType(mv::Precision::U8));

        fq->getOutputTensor(0)->setDType(getDType(mv::Precision::U8));
        fq->getOutputTensor(0)->setQuantParams(quant_paramsO);
    }
}

//NOTE: To quantize bias we use quantization parameters of input op and activations weights;
// Input-> Activation(e.g. Conv) -> Bias
// In current mcm approach it doesn't matter which qunat_params were set on bias
mv::Data::OpListIterator quantizeBias(mv::ComputationModel& model, mv::Data::OpListIterator biasOp,
                                      const mv::QuantizationParams& input_quant_params,
                                      const mv::QuantizationParams& weights_params) {
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto tensor_data = biasOp->getInputTensor(1)->getData();

    if (!input_quant_params.isScalePerTensor() && input_quant_params.getScale().size() != tensor_data.size()) {
        throw std::runtime_error("Bias and Activation quant params size mismatch");
    }

    if (!weights_params.isScalePerTensor() && weights_params.getScale().size() != tensor_data.size()) {
        throw std::runtime_error("Bias and Weights quant params size mismatch");
    }

    std::vector<int64_t> newBiasData(tensor_data.size(), 0);
    auto bias_dtype = biasOp->getInputTensor(1)->getDType();

    if (bias_dtype == getDType(mv::Precision::FP32)) {
        auto bias_data = biasOp->getInputTensor(1)->getDoubleData();

        bool needs_conversion = false;
        std::vector<int64_t> bias_zp(bias_data.size(), 0);
        std::vector<double> bias_scales;

        for (size_t i = 0; i < bias_data.size(); ++i) {
            auto activation_scale = input_quant_params.getScale(i);
            auto weights_scale = weights_params.getScale(i);

            auto bias_scale = activation_scale * weights_scale;
            bias_scales.push_back(bias_scale);
            newBiasData[i] = std::round(bias_data[i] / bias_scale);

            if (newBiasData[i] > std::numeric_limits<int32_t>::max() ||
                newBiasData[i] < std::numeric_limits<int32_t>::min()) {
                needs_conversion = true;
            }
        }

        mv::QuantizationParams bias_quant_params = {bias_zp, bias_scales, {}, {}};

        auto original_tensor = biasOp->getInputTensor(1);

        auto quantized_data = om.constantInt(om.getSourceOp(biasOp->getInputTensor(1))->getName() + ":quantized",
                                             newBiasData,
                                             original_tensor->getShape(),
                                             getDType(mv::Precision::I32),
                                             original_tensor->getOrder());
        quantized_data->setQuantParams(bias_quant_params);

        auto quantize_bias_tensor = om.bias(biasOp->getName()+":quantized",
                                            biasOp->getInputTensor(0),
                                            quantized_data);
        quantize_bias_tensor->setQuantParams(biasOp->getOutputTensor(0)->getQuantParams());

        if (biasOp->hasAttr("opId"))
        {
            unsigned currentOpId = biasOp->get<unsigned>("opId");
            om.getSourceOp(quantize_bias_tensor)->set<unsigned>("opId", currentOpId);
        }

        auto parent_op = mv::linkNewOperationsReplacement(om.getSourceOp(biasOp->getInputTensor(0)), quantize_bias_tensor, om, biasOp);

        if (needs_conversion) {
            if (isQuantizableOp(parent_op)) {
                parent_op->set<bool>("placeConversionToFloat", true);
                parent_op->set<bool>("biasOverflow", true);
            }
        }

        return findSinkLayers(dm, parent_op->getOutputTensor(0)).at(0);
    } else if (bias_dtype == getDType(mv::Precision::I32)) {
        // Do nothing
    } else {
        throw std::runtime_error("Unsupported bias data type");
    }

    return biasOp;
}

void quantizeBias(mv::ComputationModel& model) {
    mv::OpModel om(model);
    const auto biasOps = om.getOps("Bias");

    for (auto& biasOp : biasOps) {
        const auto parentOp = om.getSourceOp(biasOp->getInputTensor(0));

        if (isOpInputQuantized(om, parentOp)) {
            const auto inputQuantParams   = parentOp->getInputTensor(0)->getQuantParams();
            const auto weightsQuantParams = parentOp->getInputTensor(1)->getQuantParams();

            quantizeBias(model, biasOp, inputQuantParams, weightsQuantParams);
        }
    }
}

void removeFQ(const mv::pass::PassEntry&, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto fqOps = om.getOps("FakeQuantize");
    for (auto& fqOp : fqOps) {
        const auto parentOp = om.getSourceOp(fqOp->getInputTensor(0));

        if (hasScaleShiftFused(fqOp))
            parentOp->getOutputTensor(0)->set<bool>("scaleShiftFused", true);

        const auto inputQuantParams  = extractQuantParamsI(fqOp, true);
        const auto outputQuantParams = extractQuantParamsO(fqOp, true);
        if (parentOp->getOpType() != "Constant" &&
            parentOp->getOpType() != "ConstantInt" &&
            parentOp->getOpType() != "ConstantDataElement" &&
            !isEqual(inputQuantParams, outputQuantParams) &&
            !hasScaleShiftFused(fqOp))
            continue;

        mv::linkNewOperationsRemove(parentOp, parentOp->getOutputTensor(0), om, fqOp);
    }
}

// This is a helper function for Scale op quantization handling
// It will calculate new quant params based on Scale op scale factor
static void rescaleQuantParams(mv::QuantizationParams& quant_params, std::vector<double>& scale, size_t num_of_channels)
{
    if (scale.size() == 0 || quant_params.isEmpty() || quant_params.isInitial() || quant_params.isNeutral())
        return;
    
    // Initialize quantization params with existing ones and later rescale them
    // to take into account impact of scaling on min/max range
    auto new_min = quant_params.getMin();
    auto new_max = quant_params.getMax();
    auto new_scale = quant_params.getScale();
    auto new_zp = quant_params.getZeroPoint();

    // Check if all elements of scale data is the same
    // In such case rescale quant params based on scale factor
    if (std::equal(scale.begin() + 1, scale.end(), scale.begin()) )
    {
        double scale_factor = scale[0];
        std::for_each(new_min.begin(), new_min.end(), [scale_factor](double &el){ el /= scale_factor; });
        std::for_each(new_max.begin(), new_max.end(), [scale_factor](double &el){ el /= scale_factor; });
        std::for_each(new_scale.begin(), new_scale.end(), [scale_factor](double &el){ el /= scale_factor; });
    }
    // Check if number of elements are the same for all parameters
    // In such case perform elementwise rescaling
    else if (scale.size() == new_min.size() && scale.size() == new_max.size() && scale.size() == new_scale.size())
    {
        std::transform(new_min.begin(), new_min.end(), scale.begin(), new_min.begin(), std::divides<double>() );
        std::transform(new_max.begin(), new_max.end(), scale.begin(), new_max.begin(), std::divides<double>() );
        std::transform(new_scale.begin(), new_scale.end(), scale.begin(), new_scale.begin(), std::divides<double>() );
    }
    // Check if for quant params per tensor size of scale vector is equal to number of channels
    // In such case create new per channel quant params
    else if (quant_params.isScalePerTensor() && scale.size() == num_of_channels)
    {
        new_min.resize(num_of_channels, new_min[0]);
        new_max.resize(num_of_channels, new_max[0]);
        new_scale.resize(num_of_channels, new_scale[0]);
        new_zp.resize(num_of_channels, new_zp[0]);

        std::transform(new_min.begin(), new_min.end(), scale.begin(), new_min.begin(), std::divides<double>() );
        std::transform(new_max.begin(), new_max.end(), scale.begin(), new_max.begin(), std::divides<double>() );
        std::transform(new_scale.begin(), new_scale.end(), scale.begin(), new_scale.begin(), std::divides<double>() );
    }
    else
    {
        return;
    }

    quant_params = mv::QuantizationParams(new_zp, new_scale, new_min, new_max);
}

void quantizeScaleShift(mv::ComputationModel& model, const mv::pass::PassEntry& pass) {
    mv::OpModel om(model);
    mv::DataModel dm(model);

    // Quantize all Scale if output FQ is present
    const auto scaleOps = om.getOps("Scale");
    for (auto& scaleOp : scaleOps) {
        const auto fqOps = findOutputFakeQuantize(dm, scaleOp);
        if (fqOps.empty())
            continue;

        auto outputQuantParams = extractQuantParamsO(fqOps[0], false);
        const auto equalQuantParams = std::all_of(fqOps.begin(), fqOps.end(),
            [&outputQuantParams](const mv::Data::OpListIterator& fqOp) {
                return isEqual(outputQuantParams, extractQuantParamsO(fqOp, false));
            });
        if (!equalQuantParams) {
            pass.log(mv::Logger::MessageType::Debug, "Different output quantization params for scale op " + \
                                                      scaleOp->getName() + " - skipping quantization.");
            continue;
        }

        if (scaleOp->getInputTensor(1)->isDoubleType()) {
            auto scaleTensor = scaleOp->getInputTensor(1);
            auto originalConstOp = om.getSourceOp(scaleTensor);
            auto scaleData = scaleTensor->getDoubleData();
            size_t num_of_channels = fqOps[0]->getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

            // Since quant params will be propagated up the graph rescale
            // them based on Scale op params
            rescaleQuantParams(outputQuantParams, scaleData, num_of_channels);

            auto quantizedScaleData = std::vector<int64_t>(scaleData.size(), 127+127);
            for (size_t c = 0; c < scaleData.size(); c++) {
                if (scaleData[c] < 0) {
                    scaleData[c] = -scaleData[c];
                    quantizedScaleData[c] = 127-127;
                }
                scaleData[c] /= 127.0;
            }

            mv::QuantizationParams scalesQuantParams = {{127}, scaleData, {-inf}, {inf}};
            auto quantized_const_tensor =
                om.constantInt(originalConstOp->getName() + ":quantized",
                               quantizedScaleData, scaleTensor->getShape(),
                               getDType(mv::Precision::U8), scaleTensor->getOrder());
            quantized_const_tensor->setQuantParams(scalesQuantParams);
            if (originalConstOp->hasAttr("opId")) {
                unsigned currentOpId = originalConstOp->get<unsigned>("opId");
                quantized_const_tensor->set<unsigned>("opId", currentOpId);
                om.getSourceOp(quantized_const_tensor)->set<unsigned>("opId", currentOpId);
            }
            mv::linkNewOperationsReplacement(mv::Data::OpListIterator(), quantized_const_tensor, om, originalConstOp);

            const auto parentOp = om.getSourceOp(scaleOp->getInputTensor(0));
            if (parentOp->getOpType() != "Input" && parentOp->getOpType() != "ImplicitInput")
                propagateToParentOps(om, dm, scaleOp, getDType(mv::Precision::U8), outputQuantParams);
        }
    }
}

void fakeQuantizeConstOp(
        mv::OpModel& om,
        const mv::Data::OpListIterator& origOp,
        mv::Data::OpListIterator fqOp,
        const mv::QuantizationParams& inputQP,
        const mv::QuantizationParams& outputQP)
{
    // TODO: fp16 tensors. need to handle
    assert(origOp->getOpType() == "Constant");

    const auto origTensor = origOp->getOutputTensor(0);
    assert(origTensor->isDoubleType());

    const auto shape = origTensor->getShape();

    size_t OC, IC, KH, KW;
    getWeightsDims(shape, OC, IC, KW, KH);

    // Workaround for depthwise convolution
    // Because we expect shpae of depthwise conv to be in {KH, KW, N, 1} format
    // and all other weights have shape like {KW, KW, IC, N} we have OC=1 and IC=N
    // But further logic requires OC be equal to N and IC equal to 1
    if (origOp->hasAttr("is_depthwise_weights")) {
        std::swap(OC, IC);
    }

    auto origData = origTensor->getDoubleData();
    std::vector<double> newData(origData.size());

    const auto inputLowData = inputQP.getMin();
    const auto inputHighData = inputQP.getMax();

    const auto outputLowData = outputQP.getMin();
    const auto outputHighData = outputQP.getMax();

    // TODO: rewrite. We can do it only for ochannel loop.
    for (size_t oc = 0; oc < OC; ++oc)
    {
        const auto inputLow = inputLowData.at(inputLowData.size() > 1 ? oc : 0);
        const auto inputHigh = inputHighData.at(inputHighData.size() > 1 ? oc : 0);
        const auto inputScale = inputQP.getScale(oc);

        const auto outputLow = outputLowData.at(inputLowData.size() > 1 ? oc : 0);
        const auto outputHigh = outputHighData.at(inputHighData.size() > 1 ? oc : 0);
        const auto outputScale = outputQP.getScale(oc);

        const auto startInd = oc * IC * KW * KH;
        const auto endInd = (oc + 1) * IC * KW * KH;

        for (auto i = startInd; i < endInd; ++i) {
            const auto origVal = origData[i];

            double newVal = 0.0;
            if (origVal <= inputLow) {
                newVal = outputLow;
            } else if (origVal > inputHigh) {
                newVal = outputHigh;
            } else {
                const auto valQuant = std::round((origVal - inputLow) / inputScale);
                newVal = valQuant * outputScale + outputLow;
            }

            newData[i] = newVal;
        }
    }

    const auto newTensor = om.constant(
        origOp->getName() + ":fq",
        newData, shape,
        origTensor->getDType(),
        origTensor->getOrder());

    const auto newOp = om.getSourceOp(newTensor);

    if (origOp->hasAttr("opId"))
    {
        const auto origID = origOp->get<unsigned>("opId");
        newTensor->set<unsigned>("opId", origID);
        newOp->set<unsigned>("opId", origID);
    }

    mv::linkNewOperationsReplacement(mv::Data::OpListIterator(), newTensor, om, origOp);
    mv::linkNewOperationsRemove(newOp, newTensor, om, fqOp);
}

void fakeQuantizeConst(mv::ComputationModel& model)
{
    mv::OpModel om(model);

    for (auto& fqOp : om.getOps("FakeQuantize"))
    {
        const auto origOp = om.getSourceOp(fqOp->getInputTensor(0));

        if (origOp->getOpType() != "Constant")
        {
            continue;
        }

        const auto origTensor = origOp->getOutputTensor(0);
        if (origTensor->getDType() == mv::getDType(mv::Precision::I8) ||
            origTensor->getDType() == mv::getDType(mv::Precision::U8))
        {
            // Weights would already be quantized
            // Do nothing
            continue;
        }

        const auto inputQP = extractQuantParamsI(fqOp, false);
        const auto outputQP = extractQuantParamsO(fqOp, false);
        fakeQuantizeConstOp(om, origOp, fqOp, inputQP, outputQP);
    }
}

bool iterateThroughPassthroughOps(mv::DataModel& dm, const mv::Data::OpListIterator& opIt, const bool fallbackToFloat = false)
{
    if (fallbackToFloat) {
        opIt->getOutputTensor(0)->setDType(mv::DType("Float16"));
        opIt->getOutputTensor(0)->setQuantParams(mv::QuantizationParams::initial());
        if (opIt->getOpType()=="Concat") {
            auto inputTensors = opIt->getInputTensor();
            for (auto& tensor : inputTensors) {
               tensor->setDType(mv::DType("Float16"));
               tensor->setQuantParams(mv::QuantizationParams::initial());
            }
        }
    }

    auto childOps = mv::findSinkLayers(dm, opIt->getOutputTensor(0));
    bool checkChild = true;
    for (auto& childOp : childOps) {
        if (isOpPassthrough(childOp)) {
            if (childOp->outputSlots() && childOp->getOutputTensor(0)->isFloatingPointType())
                checkChild = false;
            if (!iterateThroughPassthroughOps(dm, childOp, fallbackToFloat))
                checkChild = false;
        } else {
            if (childOp->outputSlots() && !childOp->getOutputTensor(0)->isFloatingPointType())
                checkChild = false;
        }
    }

    return checkChild;
}

// This pass will treat cases where conversions back and forth could be avoided to improve performance.
// In particular, the following pattern will be altered:
//    [FP16] -> OP -> [U8] -> PASSTHROUGH_OPS(U8) -> [U8] -> [FP16]
// The interemdiate U8 tensors will be set to FP16 and lose their quantization information.
void reduceConversionsPatterns(mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto tensorQuantized = [](const mv::Data::TensorIterator& tensor) {
        return !tensor->isFloatingPointType() && !tensor->getQuantParams().isNeutral();
    };

    auto sortedOps = om.topologicalSort();
    for (auto& opIt : sortedOps)
    {
        if (isOpPassthrough(opIt))
            continue;

        if (!opIt->inputSlots() || !opIt->outputSlots())
            continue;

        auto inputs = opIt->getInputTensor();
        auto outputs = opIt->getOutputTensor();

        if (!(std::any_of(inputs.begin(), inputs.end(), tensorQuantized)) &&
            std::all_of(outputs.begin(), outputs.end(), tensorQuantized)) {
            if (iterateThroughPassthroughOps(dm, opIt))
                iterateThroughPassthroughOps(dm, opIt, true);
        }
    }
}

void addClamp(mv::ComputationModel& model){
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto quantizeOps = om.getOps("FakeQuantize");
    for(auto& quantizeOp : quantizeOps){
        if(quantizeOp->getName() != "g_net/enc3_2_2_1/Conv2D/fq_input_0" &&
            quantizeOp->getName() != "g_net/enc2_3_2_1/Conv2D/fq_input_0" &&
            quantizeOp->getName() != "g_net/dec2_2_2/Conv2D/fq_input_0" &&
            quantizeOp->getName() != "g_net/enc3_4_2/Conv2D/fq_input_0" &&
            quantizeOp->getName() != "g_net/enc2_3_2_2/Conv2D/fq_input_0" &&
            quantizeOp->getName() != "g_net/enc3_2_2_2/Conv2D/fq_input_0" &&
            quantizeOp->getName() != "g_net/enc3_3_2_2/Conv2D/fq_input_0" &&
            quantizeOp->getName() != "g_net/dec2_2_2_2/Conv2D/fq_input_0")
            // quantizeOp->getName() != "g_net/add_1/fq_input_1_scale_aligned" &&
            // quantizeOp->getName() != "g_net/add_1/fq_input_0" &&
            // quantizeOp->getName() != "g_net/ResizeBilinear/fq_input_0" &&
            // quantizeOp->getName() != "g_net/add_6/fq_input_0" &&
            // quantizeOp->getName() != "g_net/ResizeBilinear_1/fq_input_0")
            continue;
        
        // add clamp Operation
        auto name = quantizeOp->getName();
        auto consumerOps = mv::findSinkLayers(dm, quantizeOp->getOutputTensor(0));

        auto min_input = quantizeOp->getOutputTensor(0);
        auto min = om.minimum(name + "_new_minimum", min_input, quantizeOp->getInputTensor(2)->getDoubleData()[0]);
        // std::cout<<"######: "<<quantizeOp->getName()<<std::endl;
        // std::cout<<"******: "<<quantizeOp->getInputTensor(2)->getDoubleData()[0]<<std::endl;
        min->setQuantParams(mv::QuantizationParams::initial());
        auto minOp = om.getSourceOp(min);
        minOp->set<unsigned>("opId", quantizeOp->get<unsigned>("opId"));

        auto max = om.maximum(name + "_new_maximum", min, quantizeOp->getInputTensor(3)->getDoubleData()[0]);
        // std::cout<<"******: "<<quantizeOp->getInputTensor(3)->getDoubleData()[0]<<std::endl;
        max->setQuantParams(mv::QuantizationParams::initial());
        auto maxOp = om.getSourceOp(max);
        maxOp->set<unsigned>("opId", minOp->get<unsigned>("opId"));

        for(auto& consumerOp: consumerOps){
            std::size_t i = 0;
            for (; i < consumerOp->inputSlots(); ++i){
                if (consumerOp->getInputTensor(i)->getName() == min_input->getName())
                    break;
            }
            auto inputFlow = consumerOp.leftmostInput();
            while (inputFlow != om.flowEnd()){
                if (inputFlow->getTensor()->getName() == min_input->getName())
                    break;
                ++inputFlow;
            }
            om.undefineFlow(inputFlow);
            consumerOp->setInputTensor(max, i, false);
            om.defineFlow(max, consumerOp, i);
        }

        
    }
}

void removeInterp(mv::ComputationModel& model) {
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto interpOps = om.getOps("Interp");
    for (auto opIt:interpOps){
        if(opIt->getName()=="g_net/resize_images_4/ResizeBilinear"){
            auto inputShape = opIt->getInputTensor(0)->getShape();
            auto outputShape = opIt->getOutputTensor(0)->getShape();
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

            auto sourceTensor = parentOpIt->getOutputTensor(0);
            opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
        }
    }
}

void fuseScaleAddWithConvBiasAndPadInput(mv::ComputationModel& model) {
    mv::OpModel om(model);
    mv::DataModel dm(model);
    
    // restrict just for mobilenet v3
    auto convOps = om.getOps("Conv");
    bool flag = false;
    for (auto convOp:convOps){
      if(convOp->getName() == "MobilenetV3/Conv/BatchNorm/FusedBatchNorm/variance/Fused_Add_"){
        flag = true;
        break;
      }
    }

    // TODO how to restrict for mobilenet-v3
    if (flag){
      auto inputOps = om.getOps("Input");
      auto inputOp = inputOps[0];
      auto scaleOp = mv::findSinkLayers(dm, inputOp->getOutputTensor(0))[0];
      auto addOp = mv::findSinkLayers(dm, scaleOp->getOutputTensor(0))[0];
      auto convOp = mv::findSinkLayers(dm, addOp->getOutputTensor(0))[0];
      auto biasOp = mv::findSinkLayers(dm, convOp->getOutputTensor(0))[0];
      auto hswishOp = mv::findSinkLayers(dm, biasOp->getOutputTensor(0))[0];

      auto scaleDate = scaleOp->getInputTensor(1)->getDoubleData();
      auto addData = addOp->getInputTensor(1)->getDoubleData();
      auto weightData = convOp->getInputTensor(1)->getDoubleData();
      auto biasData = biasOp->getInputTensor(1)->getDoubleData();

      // padding Op for input
      auto inputShape = inputOp->getOutputTensor(0)->getShape();

      auto newPad = om.pad(inputOp->getName()+"_pad", inputOp->getOutputTensor(0), 
                            {0, 0, 0, 0},
                            {0, 0, 1, 1},
                            "constant",
                            -biasData[0]);
      newPad->setQuantParams(mv::QuantizationParams::initial());
      newPad->setDType(inputOp->getOutputTensor(0)->getDType());
      auto newPadOp = om.getSourceOp(newPad);
      newPadOp->set<unsigned>("opId", inputOp->get<unsigned>("opId"));

      std::vector<double> newWeightData;
      std::vector<double> newBiasData;
      // new convOp
      for (size_t i =0; i<weightData.size();i++)
        newWeightData.push_back(weightData[i]*scaleDate[0]);
      auto newWeight = om.constant(convOp->getName() + "_const", newWeightData, 
                                    convOp->getInputTensor(1)->getShape(), mv::DType("Float32"), mv::Order("NCHW"));
      newWeight->setQuantParams(convOp->getInputTensor(1)->getQuantParams());
      auto newConv = om.conv(convOp->getName()+"_new", newPad, newWeight,
                              convOp->get<std::array<unsigned short, 2>>("stride"),
                              {0, 0, 0, 0},
                              convOp->get<unsigned>("dilationFactor"),
                              convOp->get<unsigned>("group"));
      newConv->setDType(convOp->getOutputTensor(0)->getDType());
      newConv->setQuantParams(convOp->getOutputTensor(0)->getQuantParams());
      auto newWeightOp = om.getSourceOp(newWeight);
      auto newConvOp = om.getSourceOp(newConv);
      newWeightOp->set<unsigned>("opId", convOp->get<unsigned>("opId"));
      newConvOp->set<unsigned>("opId", convOp->get<unsigned>("opId"));

      // new bais Op
      for (size_t j =0;j<biasData.size();j++){
        double temp = 0;
        for (size_t k =0; k<27; k++)
          temp += weightData[j*27+k];
        temp *= addData[0];
        newBiasData.push_back(temp+biasData[j]);
      }
      auto newBiasPar = om.constant(biasOp->getName() + "_const", newBiasData, 
                                    biasOp->getInputTensor(1)->getShape(), mv::DType("Float32"), mv::Order("W"));
      newBiasPar->setQuantParams(biasOp->getInputTensor(1)->getQuantParams());
      auto newBias = om.bias(biasOp->getName() + "_new", 
                              newConvOp->getOutputTensor(0), newBiasPar);
      newBias->setQuantParams(newConvOp->getOutputTensor(0)->getQuantParams());
      newBias->setDType(biasOp->getOutputTensor(0)->getDType());
      auto newBiasParOp = om.getSourceOp(newBiasPar);
      auto newBiasOp = om.getSourceOp(newBias);
      newBiasParOp->set<unsigned>("opId", biasOp->get<unsigned>("opId"));
      newBiasOp->set<unsigned>("opId", biasOp->get<unsigned>("opId"));
      
      // set connect
      om.undefineFlow(scaleOp.leftmostInput());
      newPadOp->setInputTensor(inputOp->getOutputTensor(0), 0, false);
      
      om.undefineFlow(hswishOp.leftmostInput());
      hswishOp->setInputTensor(newBiasOp->getOutputTensor(0), 0, false);
      om.defineFlow(newBiasOp->getOutputTensor(0), hswishOp, 0);

      // remove scale and add Ops
      om.removeOp(om.getSourceOp(convOp->getInputTensor(1)));
      om.removeOp(om.getSourceOp(biasOp->getInputTensor(1)));
      om.removeOp(convOp);
      om.removeOp(biasOp);
      om.removeOp(om.getSourceOp(scaleOp->getInputTensor(1)));
      om.removeOp(om.getSourceOp(addOp->getInputTensor(1)));
      om.removeOp(scaleOp);
      om.removeOp(addOp);
    }
}

void hswishReplacement(mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);   
    auto hswishOps = om.getOps("HSwish");
    for(auto& hswishOp: hswishOps)
    {
        auto name = hswishOp->getName();
        auto inputTensor = hswishOp->getInputTensor(0);
        auto outputTensor = hswishOp->getOutputTensor(0);
        auto childOps = mv::findSinkLayers(dm, hswishOp->getOutputTensor(0));
        unsigned opId = hswishOp->get<unsigned>("opId");
        auto quantParam = inputTensor->getQuantParams();
        auto outQuantParam = outputTensor->getQuantParams();
        auto outputDtype = outputTensor->getDType();

        // Populate identity weights
        const int64_t weightsValue_i = 1;
        auto K = inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];

        std::vector<int64_t> weightsData_i(K, 0);
        for (auto i = 0u; i < K; ++i)
            weightsData_i.at(i) = weightsValue_i;
        mv::Data::TensorIterator weights_i = om.constantInt("",
                            weightsData_i,
                            {1, 1, K, 1},
                            mv::DType("UInt8"),
                            mv::Order(mv::Order::getRowMajorID(4)));
        weights_i->setQuantParams(mv::QuantizationParams({0},{1.0 / 6},{0},{255.0 / 6}));

        // Insert identity Conv
        auto identityConv = om.depthwiseConv(name + "_scale_conv", inputTensor, weights_i, {1,1}, {0, 0, 0, 0}, 1);
        identityConv->setQuantParams(mv::QuantizationParams::initial());
        auto identityConvOp = om.getSourceOp(identityConv);

        auto weightsOp_i = om.getSourceOp(weights_i);
        identityConvOp->set<unsigned>("opId", opId);
        weightsOp_i->set<unsigned>("opId", opId);

        std::vector<double> biasValue(inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 0.5);
        auto biasConst = om.constant(name + "_bias_const", biasValue, {inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float32"), mv::Order("W"));
        biasConst->setQuantParams(mv::QuantizationParams::initial());
        auto biasConstOp = om.getSourceOp(biasConst);
        biasConstOp->set<unsigned>("opId", opId);
        auto bias = om.bias(name + "_new_bias", identityConv, biasConst);
        bias->setQuantParams(mv::QuantizationParams::initial());
        auto biasOp = om.getSourceOp(bias);
        biasOp->set<unsigned>("opId", opId);

        auto min = om.minimum(name + "_new_minimum", bias, 1);
        min->setQuantParams(mv::QuantizationParams::initial());
        auto minOp = om.getSourceOp(min);
        minOp->set<unsigned>("opId", opId);

        auto max = om.maximum(name + "_new_maximum", min, 0);
        max->setQuantParams(mv::QuantizationParams::initial());
        auto maxOp = om.getSourceOp(max);
        maxOp->set<unsigned>("opId", opId);

        auto eltwiseMul = om.eltwise(name + "_new_eltwise", {inputTensor, max}, "Multiply");
        eltwiseMul->setQuantParams(outQuantParam);
        eltwiseMul->setDType(outputDtype);
        auto eltwiseMulOp = om.getSourceOp(eltwiseMul);
        eltwiseMulOp->set<unsigned>("opId", opId);

        om.removeOp(hswishOp);

        for(auto& childOp: childOps)
        {
            int idx = childOp->getOpType() == "Eltwise"? 1: 0;
            if(childOp->getName() == "MobilenetV3/expanded_conv_10/squeeze_excite/mul"
            || childOp->getName() == "MobilenetV3/expanded_conv_11/squeeze_excite/mul"
            || childOp->getName() == "MobilenetV3/expanded_conv_12/squeeze_excite/mul"
            || childOp->getName() == "MobilenetV3/expanded_conv_13/squeeze_excite/mul"
            || childOp->getName() == "MobilenetV3/expanded_conv_14/squeeze_excite/mul")
                idx = 0;
            om.defineFlow(eltwiseMul, childOp, idx);
            childOp->setInputTensor(eltwiseMul, idx, false);
        }
    }
}

void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS);
    mv::OpModel om(model);

    // fuse input scale + add into later conv + bias for mobilenet-v3 model
    fuseScaleAddWithConvBiasAndPadInput(model);

    // replace hswish with dpu layers
    hswishReplacement(model);

    if (om.getOps("FakeQuantize").empty())
        return;

    const auto globalParams = model.getGlobalConfigParams();
    const auto refMode = globalParams->hasAttr("ReferenceMode") && globalParams->get<bool>("ReferenceMode");

    if (refMode)
    {
        fakeQuantizeConst(model);

        // Mark the rest FakeQuantize ops to be executed as SW task.
        for (const auto& fqOp : om.getOps("FakeQuantize"))
        {
            fqOp->set("dType", mv::DType("Default"));
            fqOp->set("quantParams", mv::QuantizationParams({0},{1.0},{},{}));
            fqOp->getOutputTensor(0)->set("quantParams", mv::QuantizationParams({0},{1.0},{},{}));
            fqOp->set<bool>("softwareExecuted", true);
        }

        return;
    }

    // remove deblur model useless interpolate OP
    removeInterp(model);
    
    quantizeScaleShift(model, pass);

    propagateActivationParameters(model);

    quantizeConst(model);
    quantizeBias(model);

    // replace quantize-dequantize opertation with clamp Op ( min-max )
    addClamp(model);

    removeFQ(pass, model);

    reduceConversionsPatterns(model);
}
