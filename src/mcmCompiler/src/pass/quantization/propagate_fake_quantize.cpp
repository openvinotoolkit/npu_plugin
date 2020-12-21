#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/pass_quantization.hpp"
#include <math.h>

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

// This recursive function will search down for FakeQuantize operations. It will stop until either FQ is encountered
// or if it got to output or DPU task
static void getFakeQuantizeRecursiveDown(mv::Data::OpListIterator exploreOp, std::vector<mv::Data::OpListIterator>& fqOps, std::vector<bool>& fqExistStatus, mv::OpModel& om)
{
    for(auto nextOp = exploreOp.leftmostChild(); nextOp != om.opEnd(); ++nextOp)
    {
        auto opType = nextOp->getOpType();
        if (opType == "FakeQuantize")
        {
            fqOps.push_back(nextOp);
            fqExistStatus.push_back(true);
        }
        else if (nextOp->isHardwarizable())
        {
            // If we got to DPU task here than it means function didn't encounter FQ node
            // when traversing from input
            fqExistStatus.push_back(false);

        }
        else if(opType != "Output")
        {
            getFakeQuantizeRecursiveDown(nextOp, fqOps, fqExistStatus, om);
        }
    }
}

// This pass will analyze if network inputs contain also FakeQuantize operations which would allow to make
// computation in a lower precision then input precision.
// Currently it is limited in making decision if with FP32/16 input precision computation can be done
// in U8 - this will happen if FQ levels match U8 precision
static void decideComputePrecisionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS);

    mv::OpModel om(model);
    mv::DataModel dm(model);

    if (om.getOps("FakeQuantize").empty())
    {
        return;
    }

    auto networkInputs = om.getNetworkInputs();
    // Vector of FakeQuantize operations for input node
    std::vector<mv::Data::OpListIterator> inputFqOps;
    std::vector<bool> fqExistStatusVec;

    for (size_t i = 0; i < networkInputs.size(); i++)
    {
        auto inputDtype = networkInputs[i]->get<mv::DType>("dType");
        // Check network input precision. If it is already U8 then there
        // is no need to analize FQ ops as we are alrady at the lowest compute precision
        // and no need to convert from FP16/32 to U8
        if (inputDtype != mv::DType("Float16") && inputDtype != mv::DType("Float32"))
            continue;

        getFakeQuantizeRecursiveDown(networkInputs[i], inputFqOps, fqExistStatusVec, om);
    }

    // Check if all FP inputs have FQ before reaching DPU task
    for (auto fqExistStatus : fqExistStatusVec)
    {
        if (!fqExistStatus)
        {
            return;
        }
    }

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

    // Check if DPU tasks weight inputs have FQ operations which allow
    // for U8 compute precision
    auto ops = om.getOps();
    for(auto& opIt : ops)
    {
        if (!opIt->hasWeights())
            continue;

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

    // Insert Conversion from FP to U8 node after each FQ with 256 levels
    for (auto& inputFqOp : inputFqOps)
    {
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

        auto sourceFlowStart = inputFqOp.leftmostOutput();

        for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            flowsToRemove.push_back(sinkFlow);
        }

        auto outputTensor = inputFqOp->getOutputTensor(0);

        // Create a conversion layer: FQ->Conversion
        auto conversionOutput = om.conversion(om.getSourceOp(inputFqOp->getInputTensor(0))->getName() + "_convert_to_U8", inputFqOp->getOutputTensor(0), mv::DType("UInt8"));
        conversionOutput->setQuantParams(outputTensor->getQuantParams());

        conversionOps.push_back(om.getSourceOp(conversionOutput));

        // Remove previous flows: FQ->SomeOps
        for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
        {
            om.undefineFlow(flowsToRemove[flowIdx]);
        }

        // Define new flows: Conversion->SomeOps
        for(unsigned op = 0 ; op < opsToLink.size(); ++op)
        {
            opsToLink[op]->setInputTensor(conversionOutput, inputSlots[op], false);
            om.defineFlow(conversionOutput, opsToLink[op], inputSlots[op]);
        }
    }

    // Recursively update all children nodes with U8 data type
    for (auto& conversionOp : conversionOps)
    {
        updateChildrenPrecisionRecursiveDown(conversionOp, om, mv::DType("UInt8"));
    }

}

static double inf = std::numeric_limits<double>::infinity();

// This is needed to avoid a static initializer fiasco.
static const mv::QuantizationParams& initial_quant_params() {
    static mv::QuantizationParams init{{0}, {1}, {-inf}, {inf}};
    return init;
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
        return om.getSourceOp(op->getInputTensor(0))->getOpType() == "FakeQuantize";
    }

    assert(op->getInputTensor().size() > 1);
    return (om.getSourceOp(op->getInputTensor(1))->getOpType() == "FakeQuantize") ||
            op->getInputTensor(1)->getDType() == getDType(mv::Precision::U8) ||
            op->getInputTensor(1)->getDType() == getDType(mv::Precision::I8);
}

mv::QuantizationParams findOutputQuantParams(mv::ComputationModel& model, mv::Data::OpListIterator op) {
    //NOTE: If we have more than one child we have several cases
    //ALL branches without FQ -> initial quant params and float output
    //FQ only on one branch -> set this quant_params
    //FQ on all branches with same quant params ->set this quant params
    //FQ with different quant params -> assert

    mv::DataModel dm(model);
    auto idx = 0;
    auto current_ops = findSinkLayers(dm, op->getOutputTensor(0));

    while(current_ops.size() == 1 && current_ops[0]->getOpType() != "FakeQuantize" && current_ops[0]->getOpType() != "Output" && current_ops[0]->getOpType() != "Deconv") {
        idx = (current_ops[0]->getOpType() == "TopK") ? 1 : 0;
        current_ops = findSinkLayers(dm, current_ops[0]->getOutputTensor(idx));
        assert(current_ops[0]->getOutputTensor().size() < 2);
    }

    std::vector<mv::QuantizationParams> outQuantParams;
    for (size_t i = 0; i < current_ops.size(); i++) {
        if (current_ops[i]->getOpType() == "FakeQuantize") {
            outQuantParams.push_back(extractQuantParamsO(current_ops[i], op->getOpType() != "Constant"));
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

mv::QuantizationParams getParentQuantParams(mv::OpModel& /*om*/, const mv::Data::OpListIterator& op, size_t parent_idx = 0) {
    assert(op->getInputTensor().size() > 0);
    auto inputTensor = op->getInputTensor(parent_idx);
    assert(inputTensor->hasAttr("quantParams"));
    return inputTensor->get<mv::QuantizationParams>("quantParams");
}

void setQuantizationParams(mv::Data::OpListIterator& op, mv::QuantizationParams quant_params) {
    for (auto& tensor: op->getOutputTensor()) {
        tensor->set<mv::QuantizationParams>("quantParams", quant_params);
    }
}

bool areInputScalesEqual(mv::OpModel & om, mv::Data::OpListIterator op, bool zeroPointCheck) {
    std::vector<mv::QuantizationParams> input_params;
    for (size_t i = 0; i < op->getInputTensor().size(); ++i) {
        input_params.push_back(getParentQuantParams(om, op, i));
    }

    auto isScaleEqual = [](const mv::QuantizationParams& left, const mv::QuantizationParams& right) -> bool {
        return mv::isVectorsEqual(left.getScale(), right.getScale());
    };

    return std::adjacent_find(input_params.begin(), input_params.end(),
            [&](const mv::QuantizationParams& left, const mv::QuantizationParams& right) {
                return !isScaleEqual(left, right) &&
                       (zeroPointCheck || left.getZeroPoint() != right.getZeroPoint());
    }) == input_params.end();
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
void propagateParameters(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {
    mv::OpModel om(model);
    mv::QuantizationParams quant_params{{}, {}, {}, {}};
    auto sorted_ops = om.topologicalSort();
    for (auto& op : sorted_ops) {
        if (op->getOpType() == "Eltwise" || op->getOpType() == "Concat") {
            if (false == areInputScalesEqual(om, op, op->getOpType() == "Concat") )
                pass.log(mv::Logger::MessageType::Warning, "Inputs of layer type " + op->getOpType() +
                    " : " + op->getName() + " do not have the same QuantParams");
        }

        if ((isQuantizableOp(op) && isOpQuantized(om, op))
            || op->getOpType() == "Constant"
            || op->getOpType() == "ConstantDataElement" // NOTE: float16 case is not handled here
            || op->getOpType() == "Interp"
            || op->getOpType() == "Normalize" //Interp might be used for re-quantize, need the quant params
            || op->getOpType() == "Deconv") { 
            quant_params = findOutputQuantParams(model, op);

            if (op->getOpType() == "AveragePool" && isEqual(quant_params, initial_quant_params())) {
                quant_params = getParentQuantParams(om, op);
            }

            setQuantizationParams(op, quant_params);
        } else if (op->getOpType() != "Input" &&
                   op->getOpType() != "ImplicitInput" &&
                   op->getOpType() != "ImplicitInputSlice" &&
                   op->getOpType() != "ConstantInt") {
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
mv::Data::OpListIterator quantizeConstOp(mv::OpModel& om, mv::Data::OpListIterator operation,
        mv::QuantizationParams quant_paramsI, mv::QuantizationParams quant_paramsO, mv::DType precision) {
    // TODO: fp16 tensors. need to handle
    assert(operation->getOpType() == "Constant" ||
        operation->getOpType() == "ConstantDataElement");

    auto originalTensor = operation->getOutputTensor(0);
    assert(originalTensor->getDType() != getDType(mv::Precision::FP16));

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

            if (precision == getDType(mv::Precision::U8)) {
                uint8_t value = clamp<double>(new_value, 0, 255);
                quantized_weights[i] = value;
            } else if (precision == getDType(mv::Precision::I8)) {
                int8_t value = clamp<double>(new_value, -128, 127);
                quantized_weights[i] = value;
            } else {
                throw std::runtime_error("Unexpected dtype");
            }
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
            return;
        }

        auto quant_paramsI = extractQuantParamsI(fq, false);
        auto quant_paramsO = extractQuantParamsO(fq, false);
        quantizeConstOp(om, op, quant_paramsI, quant_paramsO, getDType(mv::Precision::U8));
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

    if (bias_dtype == getDType(mv::Precision::FP32)) {
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

        auto quantized_data = om.constantInt(om.getSourceOp(biasOp->getInputTensor(1))->getName() + ":quantized",
                                             newBiasData,
                                             original_tensor->getShape(),
                                             getDType(mv::Precision::I32),
                                             original_tensor->getOrder());
        quantized_data->setQuantParams(input_quant_params);

        auto quantize_bias_tensor = om.bias(biasOp->getName()+":quantized",
                                            biasOp->getInputTensor(0),
                                            quantized_data);
        quantize_bias_tensor->setDType(quantized_data->getDType());
        quantize_bias_tensor->setQuantParams(input_quant_params);

        auto parent_op = mv::linkNewOperationsReplacement(om.getSourceOp(biasOp->getInputTensor(0)), quantize_bias_tensor, om, biasOp);
        return findSinkLayers(dm, parent_op->getOutputTensor(0)).at(0);
    } else if (bias_dtype == getDType(mv::Precision::I32)) {
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

            auto weights_params = activationOp->getInputTensor(1)->getQuantParams();

            quantizeBias(model, biasOp, input_quant_params, weights_params);
        }
    }
}

void quantizeIO(mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto inputs = om.getOps("Input");
    auto implicit_inputs = om.getOps("ImplicitInput");
    inputs.insert(inputs.end(), implicit_inputs.begin(), implicit_inputs.end());
    mv::DataModel dm(om);
    for (size_t idx = 0; idx < inputs.size(); idx++) {
        auto input = inputs.at(idx);
        auto current_ops = findSinkLayers(dm, input->getOutputTensor(0));

        mv::QuantizationParams inputQuantParams = input->getOutputTensor(0)->getQuantParams();
        if(current_ops.size() == 1 && current_ops[0]->getOpType() == "FakeQuantize") {
            inputQuantParams = extractQuantParams(current_ops[0], input->getOpType() != "Constant");
        }

        std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
        bool scaleFuseInput = globalParams->hasAttr("ScaleFuseInput") ? globalParams->get<bool>("ScaleFuseInput") : false;

        //Fuse scaleshift(multiply+add) into quantization parameters to boost performance
        if(current_ops.size() == 1 && current_ops[0]->getOpType() == "Scale" && scaleFuseInput) {
            auto child_ops = findSinkLayers(dm, current_ops[0]->getOutputTensor(0));
            if(child_ops.size() == 1 && child_ops[0]->getOpType() == "Bias") {
                auto next_child_ops = findSinkLayers(dm, child_ops[0]->getOutputTensor(0));
                if(next_child_ops.size() == 1 && next_child_ops[0]->getOpType() == "FakeQuantize") {
                    auto inputC = input->getOutputTensor(0)->getShape()[2];

                    if (current_ops[0]->getInputTensor(1)->isFloatingPointType()) {
                        std::runtime_error("Unsupported fuse scaleshift DType");
                    }
                    std::vector<int64_t> scaleData = current_ops[0]->getInputTensor(1)->getIntData();
                    auto biasData = child_ops[0]->getInputTensor(1)->getIntData();
                    std::vector<double> realScaleValue(inputC);
                    std::vector<double> realBiasValue(inputC);

                    //calculate real scale values
                    auto scaleDataQuantParams = current_ops[0]->getInputTensor(1)->getQuantParams();
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
                    auto biasDataQuantParams = child_ops[0]->getInputTensor(1)->getQuantParams();
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

                    // TODO: Input scale fusing cannot be performed if per channel parameters (scale, bias)
                    // are very different from each other. Add conditional code which will check
                    // similarity of elements of realBiasValue and realScaleValue

                    float output_min_value = (realBiasValue[0] * realScaleValue[0] +
                    realBiasValue[1] * realScaleValue[1] + realBiasValue[2] * realScaleValue[2]) / 3;

                    float output_max_value = ((realBiasValue[0] + 255) * realScaleValue[0] +
                    (realBiasValue[1] + 255) * realScaleValue[1] + (realBiasValue[2] + 255) * realScaleValue[2]) / 3;

                    zero_points.push_back(mv::calculateZeroPoint(output_min_value,
                        output_max_value, getDType(mv::Precision::U8), levels));
                    scales.push_back(mv::calculateScales(output_min_value,
                        output_max_value, levels));
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
    setQuantizationParams(output, mv::QuantizationParams::empty());
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

            setQuantizationParams(current_op, findOutputQuantParams(om, current_op));

            // Quantize input bias
            // olny if scale-fuse starts
            std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
            if(globalParams->hasAttr("ScaleFuseInput") ? globalParams->get<bool>("ScaleFuseInput") : false){
                current_op = findSinkLayers(dm, current_op->getOutputTensor(0)).at(0);
                if (current_op->getOpType() == "Bias" && current_op->getInputTensor(1)->isFloatingPointType()) {
                    auto bias_op = quantizeBias(model, current_op, initial_quant_params(), scalesQuantParams);
                    setQuantizationParams(bias_op, getParentQuantParams(om, bias_op));
                }
            }
        }
    }
}

void fakeQuantizeConstOp(
        mv::OpModel& om,
        mv::Data::OpListIterator origOp,
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
            return;
        }

        const auto inputQP = extractQuantParamsI(fqOp, false);
        const auto outputQP = extractQuantParamsO(fqOp, false);
        fakeQuantizeConstOp(om, origOp, fqOp, inputQP, outputQP);
    }
}

void quantizeGraphFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS);

    mv::OpModel om(model);

    if (om.getOps("FakeQuantize").empty())
    {
        return;
    }

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

    quantizeInputScaleShift(model);
    quantizeIO(model);

    propagateParameters(pass, model);

    quantizeConst(model);
    quantizeBias(model);

    removeFQ(pass, model);
}
