#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/pass_quantization.hpp"

static void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);
static void decideComputePrecisionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);


namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(DecideComputePrecision)
        .setFunc(decideComputePrecisionFcn)
        .setDescription(
            "This pass checks FQ node after input and makes decision on internal network compute precision"
        );

        MV_REGISTER_PASS(FakeQuantize)
        .setFunc(quantizeGraphFcn)
        .setDescription(
            "This pass propagate parameters from FakeQuantize ops to all other operations and quantize const data"
        );
    }
}

// This recursive function will traverse down and update each op output tensor precision to destination type
static void updateChildrenPrecisionRecursiveDown(mv::Data::OpListIterator exploreOp, mv::OpModel& om, mv::DType dataType)
{
    for(auto nextOp = exploreOp.leftmostChild(); nextOp != om.opEnd(); ++nextOp)
    {
        auto opType = nextOp->getOpType();
        if(opType != "Output" && opType != "Constant")
        {
            auto outputTensors = nextOp->getOutputTensor();

            // If precision is already properly set skip this path
            if (outputTensors[0]->getDType() == dataType)
            {
                continue;
            }

            for (auto& outputTensor : outputTensors)
            {
                outputTensor->setDType(dataType);
            }
            updateChildrenPrecisionRecursiveDown(nextOp, om, dataType);
        }
    }
}

// This recursive function will traverse up and update each tensor precision to destination type
static void updateParentPrecisionRecursiveUp(mv::Data::OpListIterator exploreOp, mv::OpModel& om, mv::DType dataType)
{
    auto inputTensor = exploreOp->getInputTensor(0);
    auto parentOp = om.getSourceOp(inputTensor);
    inputTensor->setDType(dataType);
    inputTensor->set<bool>("FPConversionPath", true);
    parentOp->set<mv::DType>("dType", dataType);
    if (parentOp->getOpType() != "Input")
    {
        updateParentPrecisionRecursiveUp(parentOp, om, dataType);
    }
}

// This recursive function will search down for FakeQuantize operations. It will stop until either FQ is encountered
// or if it got to output or DPU task
static void getFakeQuantizeRecursiveDown(mv::Data::OpListIterator exploreOp, std::vector<mv::Data::OpListIterator>& fqOps, std::vector<mv::Data::OpListIterator>& notQuantizedDpuTasks, mv::OpModel& om)
{
    for(auto nextOp = exploreOp.leftmostChild(); nextOp != om.opEnd(); ++nextOp)
    {
        auto opType = nextOp->getOpType();
        if (opType == "FakeQuantize")
        {
            fqOps.push_back(nextOp);
        }
        else if (opType == "Output")
        {
            continue;
        }
        else
        {
            if (nextOp->isHardwarizable())
                notQuantizedDpuTasks.push_back(nextOp);
            getFakeQuantizeRecursiveDown(nextOp, fqOps, notQuantizedDpuTasks, om);
        }
    }
}

// This pass will analyze if network inputs contain also FakeQuantize operations which would allow to make
// computation in a lower precision then input precision.
// Currently it is limited in making decision if with FP32/16 input precision computation can be done
// in U8 - this will happen if FQ levels match U8 precision
static void decideComputePrecisionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& /* compilationDescriptor*/, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS);

    mv::OpModel om(model);
    mv::DataModel dm(model);

    if (om.getOps("FakeQuantize").empty())
    {
        return;
    }

    auto networkInputs = om.getNetworkInputs();

    // This pass 'DecideComputePrecision' is only a WA to make FP-input SuperResolution pass. 
    // Will be removed after ticket CVS-40814 being merged.
    // This condition is to avoid regression on aclnet network.
    if (networkInputs.size() < 2)
        return;
    
    // Vector of FakeQuantize operations for input node
    std::vector<mv::Data::OpListIterator> inputFqOps;
    std::vector<mv::Data::OpListIterator> notQuantizedDpuTasks;
    bool inputIsU8 = true;

    for (size_t i = 0; i < networkInputs.size(); i++)
    {
        auto inputDtype = networkInputs[i]->get<mv::DType>("dType");
        // Check network input precision. If it is already U8 then there
        // is no need to analize FQ ops as we are alrady at the lowest compute precision
        // and no need to convert from FP16/32 to U8
        if (inputDtype != mv::DType("Float16") && inputDtype != mv::DType("Float32"))
            continue;

        inputIsU8 = false;
        getFakeQuantizeRecursiveDown(networkInputs[i], inputFqOps, notQuantizedDpuTasks, om);
    }

    // If all the input is U8 no need for any transition
    if (inputIsU8)
        return;

    // Check if all input paths have FQ with 256 levels that allow to change precision to U8
    unsigned fqLevels;
    for (auto& inputFqOp : inputFqOps)
    {
        // Check FQ op levels. If it is above 256 then we cannot convert to U8
        fqLevels = inputFqOp->get<unsigned>("levels");
        if (fqLevels > 256)
        {
            pass.log(mv::Logger::MessageType::Warning, "Input is served by FakeQuantize " + inputFqOp->getName() + " which has " +
                std::to_string(fqLevels) + " levels - not supported case for transition to U8 compute precision");
            return;
        }
    }

    // Check if DPU tasks that will operate on U8 have weight inputs with FQ operation for U8 precision
    auto ops = om.getOps();
    for(auto& opIt : ops)
    {
        if (!opIt->hasWeights())
            continue;

        // Check if an op has already been identified as a not quantized dpu task
        if (std::find(notQuantizedDpuTasks.begin(), notQuantizedDpuTasks.end(), opIt) != notQuantizedDpuTasks.end())
        {
            continue;
        }

        auto weightTensor = opIt->getInputTensor(1);
        if (weightTensor->getDType() == mv::DType("UInt8"))
        {
            // If weight tensor is already U8 then no need to check FQ op
            continue;
        }

        auto parentOp = om.getSourceOp(weightTensor);
        std::string parentOpType;
        // Analyze weight input branch and check it if has any FQ node
        while(true)
        {
            if (!parentOp)
                return;

            parentOpType = parentOp->getOpType();
            if (parentOpType == "FakeQuantize")
            {
                // If there are more than 256 levels of quantization
                // then it means we cannot convert to U8, thus exit early
                fqLevels = parentOp->get<unsigned>("levels");
                if (fqLevels > 256)
                {
                    pass.log(mv::Logger::MessageType::Warning, "Weight input is served by FakeQuantize " + parentOp->getName() + " which has " +
                        std::to_string(fqLevels) + " levels - not supported case for transition to U8 compute precision");
                    return;
                }
                // Else move to next DPU task and stop analyzing this branch
                else
                {
                    break;
                }
            }
            else if (parentOpType == "Constant")
            {
                pass.log(mv::Logger::MessageType::Warning, "Weight input " + parentOp->getName() +
                " doesn't have FakeQuantize operation - not supported case for transition to U8 compute precision");
                return;
            }
            parentOp = parentOp.leftmostParent();
        }
    }


    // Store all conversion ops in a vector. Later it will be used as
    // a starting point for data type update
    std::vector<mv::Data::OpListIterator> conversionOps;

    // Insert Conversion from FP to U8 node before each FQ with 256 levels
    for (auto& inputFqOp : inputFqOps)
    {
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

        // Input to FQ will become new input for Conversion layer
        auto inputTensor = inputFqOp->getInputTensor(0);

        auto sourceFlowStart = om.getSourceOp(inputTensor).leftmostOutput();

        for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            flowsToRemove.push_back(sinkFlow);
        }

        // Create a conversion layer (SomeOp->Conversion->FQ) and update
        // tensors quant params so that later passes will correctly propagate quant settings
        // and proper Scale and ZeroPoint will be set for Conversion UPA task
        mv::QuantizationParams conversionInputQuantParams = extractQuantParamsI(inputFqOp, 1);
        mv::QuantizationParams conversionOutputQuantParams = extractQuantParamsO(inputFqOp, 1);
        auto conversionOutput = om.conversion(om.getSourceOp(inputTensor)->getName() + "_convert_to_U8", inputTensor, mv::DType("UInt8"));
        conversionOutput->setQuantParams(conversionOutputQuantParams);
        inputTensor->setQuantParams(conversionInputQuantParams);

        auto conversionOp = om.getSourceOp(conversionOutput);
        conversionOp->set("scale", 1.0 / conversionInputQuantParams.getScale()[0]);
        conversionOp->set("bias", conversionInputQuantParams.getZeroPoint()[0]);

        conversionOps.push_back(conversionOp);

        // Remove previous flows: SomeOps->FQ
        for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
        {
            om.undefineFlow(flowsToRemove[flowIdx]);
        }

        // Define new flows: Conversion->FQ
        for(unsigned op = 0 ; op < opsToLink.size(); ++op)
        {
            opsToLink[op]->setInputTensor(conversionOutput, inputSlots[op], false);
            om.defineFlow(conversionOutput, opsToLink[op], inputSlots[op]);
        }

        if (om.getSourceOp(inputTensor)->getOpType() == "ImplicitInput") {
            updateParentPrecisionRecursiveUp(conversionOp, om, inputTensor->getDType());
        }
    }

    // Recursively update all children nodes with U8 data type
    for (auto& conversionOp : conversionOps)
    {
        updateChildrenPrecisionRecursiveDown(conversionOp, om, mv::DType("UInt8"));
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
    static std::set<std::string> quantizable_ops{"Conv", "FullyConnected", "Eltwise" , "AveragePool", "DepthwiseConv", "Scale"};
    return quantizable_ops.count(op->getOpType());
}

bool isOpQuantized(mv::OpModel& om, const mv::Data::OpListIterator& op) {
    if (!isQuantizableOp(op)) {
        return false;
    }

    if (op->getOpType() == "AveragePool") {
        return om.getSourceOp(op->getInputTensor(0))->getOpType() == "FakeQuantize";
    }

    assert(op->getInputTensor().size() > 1);
    return (om.getSourceOp(op->getInputTensor(1))->getOpType() == "FakeQuantize") ||
            op->getInputTensor(1)->getDType() == getDType(mv::Precision::U8) ||
            op->getInputTensor(1)->getDType() == getDType(mv::Precision::I8);
}

bool isOpPassthrough(const mv::Data::OpListIterator& op)
{
    std::vector<std::string> passthroughOps = {
        "Bias", "Relu", "LeakyRelu", "Concat", "Maximum", "Minimum", "ReorgYolo", "Reshape", "Permute", "Interp", "Resample", "MaxPool", "Mish", "Sigmoid", "Tanh", "HSwish"
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

        const auto inputQuantParams  = extractQuantParamsI(fqOp, true);
        const auto outputQuantParams = extractQuantParamsO(fqOp, true);
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

        if (isOpQuantized(om, parentOp)) {
            const auto inputQuantParams   = parentOp->getInputTensor(0)->getQuantParams();
            const auto weightsQuantParams = parentOp->getInputTensor(1)->getQuantParams();

            quantizeBias(model, biasOp, inputQuantParams, weightsQuantParams);
        }
    }
}

void quantizeIO(mv::ComputationModel& model) {
    mv::OpModel om(model);
    mv::DataModel dm(om);

    auto inputOps = om.getOps("Input");
    auto implicitInputOps = om.getOps("ImplicitInput");
    inputOps.insert(inputOps.end(), implicitInputOps.begin(), implicitInputOps.end());

    for (auto& inputOp : inputOps) {
        auto inputTensor = inputOp->getOutputTensor(0);
        auto consumerOps = findSinkLayers(dm, inputTensor);
        for (auto& consumerOp : consumerOps) {
            if (consumerOp->getOpType() != "FakeQuantize")
                continue;

            const auto inputQuantParams  = extractQuantParamsI(consumerOp, true);
            const auto outputQuantParams = extractQuantParamsO(consumerOp, true);

            if (inputOp->get<mv::DType>("dType") != getDType(mv::Precision::U8)) {
                auto convertedTensor = om.conversion(inputOp->getName() + "_convert", inputTensor, getDType(mv::Precision::U8));
                inputTensor->setQuantParams(inputQuantParams);
                convertedTensor->setQuantParams(outputQuantParams);

                auto conversionOp = om.getSourceOp(convertedTensor);
                conversionOp->set<double>("scale", 1.0 / inputQuantParams.getScale()[0]);
                conversionOp->set<int64_t>("bias", inputQuantParams.getZeroPoint()[0]);
                if (consumerOp->hasAttr("opId"))
                    conversionOp->set<unsigned>("opId", consumerOp->get<unsigned>("opId"));

                mv::linkNewMultipleOperationsReplacement(inputOp, {convertedTensor}, om, consumerOp);
            } else {
                mv::linkNewOperationsRemove(inputOp, inputTensor, om, consumerOp);
                inputTensor->setDType(getDType(mv::Precision::U8));
                inputTensor->setQuantParams(outputQuantParams);
            }
        }
    }
}

void removeFQ(const mv::pass::PassEntry&, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto fqOps = om.getOps("FakeQuantize");
    for (auto& fqOp : fqOps) {
        const auto parentOp = om.getSourceOp(fqOp->getInputTensor(0));

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

void quantizeScaleShift(mv::ComputationModel& model, const mv::pass::PassEntry& pass) {
    mv::OpModel om(model);
    mv::DataModel dm(model);

    // Quantize all Scale if output FQ is present
    const auto scaleOps = om.getOps("Scale");
    for (auto& scaleOp : scaleOps) {
        const auto fqOps = findOutputFakeQuantize(dm, scaleOp);
        if (fqOps.empty())
            continue;

        const auto outputQuantParams = extractQuantParamsO(fqOps[0], false);
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
    }

    auto childOps = mv::findSinkLayers(dm, opIt->getOutputTensor(0));
    for (auto& childOp : childOps) {
        if (isOpPassthrough(childOp)) {
            if (childOp->outputSlots() && childOp->getOutputTensor(0)->isFloatingPointType())
                return false;
            if (!iterateThroughPassthroughOps(dm, childOp, fallbackToFloat))
                return false;
        } else {
            if (childOp->outputSlots() && !childOp->getOutputTensor(0)->isFloatingPointType())
                return false;
        }
    }

    return true;
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

        if (!tensorQuantized(opIt->getInputTensor(0)) && tensorQuantized(opIt->getOutputTensor(0))) {
            if (iterateThroughPassthroughOps(dm, opIt))
                iterateThroughPassthroughOps(dm, opIt, true);
        }
    }
}

void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS);

    mv::OpModel om(model);

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

    quantizeScaleShift(model, pass);
    quantizeIO(model);

    propagateActivationParameters(model);

    quantizeConst(model);
    quantizeBias(model);

    removeFQ(pass, model);

    reduceConversionsPatterns(model);
}
