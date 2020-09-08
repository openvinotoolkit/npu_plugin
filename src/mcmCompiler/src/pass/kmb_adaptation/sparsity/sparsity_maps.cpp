#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "mcm/utils/custom_math.hpp"
#include <math.h>

static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void generateSparsityMapsUnpopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void generateSparsityMapsEltwiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void setSparsityAttrForUnpopulatedFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateSparsityMapsPopulatedTensors)
        .setFunc(generateSparsityMapsPopulatedTensorsFcn)
        .setDescription(
            "Generates sparsity maps for populated tensors."
        );

        MV_REGISTER_PASS(GenerateSparsityMapsUnpopulatedTensors)
        .setFunc(generateSparsityMapsUnpopulatedTensorsFcn)
        .setDescription(
            "Generates sparsity maps for unpopulated tensors."
        );

        MV_REGISTER_PASS(GenerateSparsityMapsEltwise)
        .setFunc(generateSparsityMapsEltwiseFcn)
        .setDescription(
            "Generates sparsity maps for unpopulated tensors involved in eltwise operations."
        );

        MV_REGISTER_PASS(SetSparsityAttrForUnpopulatedTensors)
        .setFunc(setSparsityAttrForUnpopulatedFnc)
        .setDescription(
            "sets needs sparsity attr for unpopulated tensors."
        );
    }
}

mv::Data::TensorIterator createFakeSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, const mv::Shape& sparsityShape, const std::vector<int64_t>& sparsityMapData)
{
    auto sparsityMap = om.constantInt(sparsityMapData, sparsityShape, mv::DType("UInt8"), mv::Order("NHWC"), {{},{},{},{}},sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    unsigned newSize = dpuTaskOp->addInputTensor(sparsityMap);
    om.defineFlow(sparsityMap, dpuTaskOp, newSize - 1);

    return sparsityMap;
}


std::vector<int8_t> createBitPattern(uint16_t kernelW, uint16_t kernelH, uint16_t windowsSize, uint16_t inputChannels)
{
    std::vector<int8_t> bitpattern;
    bitpattern.reserve(windowsSize*kernelH*inputChannels);
    for(size_t c = 0; c < inputChannels; c++)
        for(size_t y = 0; y < kernelH; y++)
            for(size_t x = 0; x < windowsSize; x++)
                if (x < kernelW)
                    bitpattern.emplace_back(1);
                else
                    bitpattern.emplace_back(0);
    return bitpattern;
}

std::size_t predictSubTensorShape(std::size_t heightShape, unsigned int numClusters)
{
    return ceil(heightShape/numClusters);
}

// The sparsity maps relative to populated tensors have to be generated BEFORE the dma passes.
// As they have to be DMAed into CMX.
static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto globalParams = model.getGlobalConfigParams();
    unsigned int numClusters = (unsigned int)globalParams->get<int>("Number_of_Clusters");

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            bool weightsSparsity = dpuTask->hasAttr("weightsSparsity") ? dpuTask->get<bool>("weightsSparsity") : false;
            std::string taskOp = dpuTask->get<std::string>("taskOp");
            pass.log(mv::Logger::MessageType::Debug, " taskOp "  + dpuTask->get<std::string>("taskOp"));
            bool isChannelMajorConv = taskOp == "ChannelMajorConvolution";
            bool isPooling = taskOp == "MaxPool";
            bool isDepthWiseConv = taskOp == "DepthwiseConv";

            // NOTE by Marco: Checking for Elementwise is necessary, as it is the only operation
            // that does not support neither Real Sparsity or Fake Sparsity (aka Activation Window)
            // being it the hackiest operation ever
            bool isElementWise = taskOp == "Eltwise";

            //for max pooling and deptwise convolution (and CM conv, if enabled) we need to
            //generate sparsity data even if those layers do not support sparsity.
            if (isPooling || isDepthWiseConv || isChannelMajorConv)
            {
                uint16_t kernelW, kernelH;

                auto strides = dpuTask->get<std::array<unsigned short, 2>>("stride");
                auto inputChannels = dpuTask->getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                auto outputChannels = dpuTask->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

                // Using the check in this way, instead of on the operation type
                // makes this pass work on both aligned and unaligned convolutions.
                if (dpuTask->hasAttr("kSize"))
                {
                    auto kernelShape = dpuTask->get<std::array<unsigned short, 2>>("kSize");
                    kernelW = kernelShape[0];
                    kernelH = kernelShape[1];
                }
                else
                {
                    auto weightsShape = dpuTask->getInputTensor(1)->getShape();
                    kernelW = weightsShape[mv::KERNEL_WIDTH];
                    kernelH = weightsShape[mv::KERNEL_HEIGHT];
                }

                mv::DType dataType = mv::DType("UInt8");

                if (isPooling || isDepthWiseConv || isChannelMajorConv)
                {
                    if (dpuTask->hasAttr("floatPrecision") && dpuTask->get<bool>("floatPrecision"))
                    {
                        dataType = mv::DType("Float16");
                    }
                }

                auto windowsSize = getWindowSize(kernelW, strides[0], dataType);

                pass.log(mv::Logger::MessageType::Debug, "windowSize " + std::to_string(windowsSize));
                pass.log(mv::Logger::MessageType::Debug, "OutputChannels " + std::to_string(outputChannels));

                std::vector<int8_t> bitpattern;
                std::vector<uint8_t> perChannelSparsity;
                std::vector<std::size_t> ndims(4);
                if (isPooling || isDepthWiseConv)
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, 1));
                    perChannelSparsity.resize(static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)) * 16);//allocate once
                    ndims = {16 * static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)), 1, 1, inputChannels};
                }
                else //isChannelMajorConvolution
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, inputChannels));
                    auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
                    auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));
                    perChannelSparsity.resize(NumberOfRowsSparistyBytes * 16);//allocate once
                    ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
                }

                int channelLenght = bitpattern.size();

                //populate sparsity
                pass.log(mv::Logger::MessageType::Debug, " perChannelSize = " + std::to_string(perChannelSparsity.size()) );
                for (size_t i = 0; i < bitpattern.size(); i++)
                {
                    //pass.log(mv::Logger::MessageType::Debug, " i = " + std::to_string(i));
                    //pass.log(mv::Logger::MessageType::Debug, " perChannelIDx = " + std::to_string(((i/128)*16 + (i%128)/8)) );
                    perChannelSparsity.at((i/128)*16 + (i%128)/8) |= bitpattern[i] << (i%8); //use at to check boundaries - just in case
                }

                //Create Tensor with Sparsity Data
                mv::Shape sparsityShape(ndims);
                std::vector<int64_t> data(sparsityShape.totalSize(), 0);
                //mv::Tensor sparsityTensor("backup", sparsityShape, mv::DType("UInt8"), mv::Order("WHCN"), data);
                auto sparsityTensor = mv::Tensor(dpuTask->getName() + "_sparse_dw", sparsityShape, mv::DType("UInt8"), mv::Order("NHWC"), data);

                // NOTE: This loop can probably be simplified without using an auxiliary tensor
                for(unsigned kx = 0; kx < sparsityShape[0]; ++kx)
                    for(unsigned ky = 0; ky < sparsityShape[1]; ++ky)
                        for(unsigned ic = 0; ic < sparsityShape[2]; ++ic)
                            for(unsigned oc = 0; oc < sparsityShape[3]; ++oc)
                                sparsityTensor.at({kx, ky, ic, oc}) = static_cast<int64_t>(perChannelSparsity[ky*sparsityShape[0] + kx]);

                std::string opName = dpuTask->getName();

                auto fakeSparsityMap = createFakeSparsityMap(om, dpuTask, mv::createFakeSparsityMapName(opName), sparsityShape, sparsityTensor.getIntData());
                fakeSparsityMap->set<int>("channelLength", channelLenght);

                dpuTask->set<bool>("fakeSparsity", true);
                dpuTask->set<size_t>("fakeSparsityIndex", dpuTask->inputSlots()-1);
            }
            else if(weightsSparsity && !isElementWise)
            {
                // Here only in the case of ZMajorConvolution
                auto weightsTensor = dpuTask->getInputTensor(1);

                // NOTE: Facultative, but doesn't cause overload
                weightsTensor->setOrder(mv::Order("NHWC"));

                if(weightsTensor->setSparse())
                    dm.defineTensor(weightsTensor->getSparsityMap());
            }
            //NOTE: Here is handled a specific case and this is why is treated seperately

            if (dpuTask->isSparsityConsumer() &&
                dpuTask->hasAttr("activationSparsityCompilerSolving") &&
                dpuTask->get<bool>("activationSparsityCompilerSolving"))
            {
                std::vector<mv::Data::TensorIterator> activationTensors;
                if (dpuTask->get<std::string>("taskOp") == "Conv")
                {
                    activationTensors.push_back(dpuTask->getInputTensor(0));
                }
                else if (dpuTask->get<std::string>("taskOp") == "Eltwise")
                {
                    activationTensors.push_back(dpuTask->getInputTensor(0));
                    activationTensors.push_back(dpuTask->getInputTensor(1));
                }

                for (size_t tidx = 0; tidx < activationTensors.size(); tidx++)
                {
                    auto inputTensor  = activationTensors[tidx];

                    size_t w = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION], h = 0,
                            c = inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];

                    if (std::find(weightSegmentableStrategies.begin(), weightSegmentableStrategies.end(), dpuTask->get<std::string>("splitStrategy"))
                            != weightSegmentableStrategies.end())
                    {
                        h = inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION];
                    }
                    //NOTE: SoH, HKSwitch
                    else if (std::find(activationSegmentableStrategies.begin(), activationSegmentableStrategies.end(), dpuTask->get<std::string>("splitStrategy"))
                             != activationSegmentableStrategies.end())
                    {
                        h = predictSubTensorShape(inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION], numClusters);
                    }
                    // Sparse map has to be contiguously alligned at 16 bytes for first (N - 1) clusters
                   // For the A0 bug impacting SOH & kernel > 1 compiler generated activation sparsity is the workaround
                   // For activations where the height is not divisible by 4 for example 149x149x32, then we need to adjust
                   // The size of the sparsity map so that is alligned at 16 bytes for first (N - 1) clusters

                    while ((w*h*c)%128 != 0)
                       w+=1;

                    //every element of sparsity map describes 8 elements of normal tensor
                    auto mapShape = mv::Shape({w,
                                            {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                                            {inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION]/8},
                                            {1}});
                    std::vector<int64_t> unpopulatedSparsityMapData(mapShape.totalSize(), 255);
                    mv::QuantizationParams quantParams = {{},{},{},{}};
                    std::string unpopulatedSparsityMapName = dpuTask->getName() +
                        "activation_map_" + std::to_string(tidx);
                    auto unpopulatedSparsityMap =
                        om.constantInt(unpopulatedSparsityMapData, mapShape,
                                    mv::DType("UInt8"), mv::Order("NHWC"),
                                    quantParams, unpopulatedSparsityMapName);
                    unpopulatedSparsityMap->set<bool>("solvedSparsity", true);
                    om.getSourceOp(unpopulatedSparsityMap)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                    unsigned newInputsSize = dpuTask->addInputTensor(unpopulatedSparsityMap);
                    om.defineFlow(unpopulatedSparsityMap, dpuTask, newInputsSize - 1);
                    auto smTensorIdx = dpuTask->hasAttr("unpopulatedSparsityMapIndex") ?
                        dpuTask->get<std::vector<size_t>>("unpopulatedSparsityMapIndex") :
                        std::vector<size_t>();
                    smTensorIdx.push_back(newInputsSize - 1);
                    dpuTask->set<std::vector<size_t>>("unpopulatedSparsityMapIndex", smTensorIdx);

                    mv::Shape storageElementShape =
                        mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
                            {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                            {1},
                            {1}});
                    std::vector<int64_t> storageElementData(storageElementShape.totalSize(), 0);
                    std::string storageElementName = dpuTask->getName() +
                        "storage_element_map" + std::to_string(tidx);
                    auto storageElement =
                        om.constantInt(storageElementData, storageElementShape,
                                    mv::DType("Int32"), mv::Order("NHWC"),
                                    quantParams, storageElementName);
                    storageElement->set<bool>("solvedSparsity", true);
                    om.getSourceOp(storageElement)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                    newInputsSize = dpuTask->addInputTensor(storageElement);
                    om.defineFlow(storageElement, dpuTask, newInputsSize - 1);
                    auto seTensorIdx = dpuTask->hasAttr("storageElementIndex") ?
                        dpuTask->get<std::vector<size_t>>("storageElementIndex") :
                        std::vector<size_t>();
                    seTensorIdx.push_back(newInputsSize - 1);
                    dpuTask->set<std::vector<size_t>>("storageElementIndex", seTensorIdx);
                }
            }
            // Here we generate the sparsity maps required for dilated convolution
            // Should the sparsity map be size of the full input tensor or just the "sub conv" input tensor ?
            // I think it should be the full input tensor size ?
            if (dpuTask->hasAttr("activationSparsityCompilerSolvingForDilatedConv") && dpuTask->get<bool>("activationSparsityCompilerSolvingForDilatedConv"))
            {
                auto inputTensorShape = dpuTask->getInputTensor(0)->getShape();
                //every element of sparsity map describes 8 elements of normal tensor
                //TODO re-use sparsity map if possible?
                // if the sparsity map should only be the size of the "sub conv input tensor" then change this in future
                auto mapShape = mv::Shape({{inputTensorShape[mv::IO_WIDTH_DIMENSION]},
                                           {inputTensorShape[mv::IO_HEIGHT_DIMENSION]},
                                           {inputTensorShape[mv::IO_CHANNEL_DIMENSION]/8},
                                           {1}});

                std::vector<int64_t> unpopulatedSparsityMapData(mapShape.totalSize(), 255); // 255 converts to all 1's in SM
                mv::QuantizationParams quantParams = {{},{},{},{}};
                std::string unpopulatedSparsityMapName = dpuTask->getName() + "activation_map";
                auto unpopulatedSparsityMap = om.constantInt(unpopulatedSparsityMapData, mapShape, mv::DType("UInt8"), mv::Order("NHWC"), quantParams, unpopulatedSparsityMapName);
                om.getSourceOp(unpopulatedSparsityMap)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                unsigned newInputsSize = dpuTask->addInputTensor(unpopulatedSparsityMap);
                unpopulatedSparsityMap->set<bool>("dilatedSubConvSM", true);
                om.defineFlow(unpopulatedSparsityMap, dpuTask, newInputsSize - 1);
                auto smTensorIdx = dpuTask->hasAttr("unpopulatedSparsityMapIndex") ?
                        dpuTask->get<std::vector<size_t>>("unpopulatedSparsityMapIndex") :
                        std::vector<size_t>();
                smTensorIdx.push_back(newInputsSize - 1);
                dpuTask->set<std::vector<size_t>>("unpopulatedSparsityMapIndex", smTensorIdx);

                // Here we generate a storage element pointer table of all 0's for the dilated conv case
                // The logic to generate SEPs for dilated conv should be added in weight_tables.cpp - function populateActivationStorageElementMapForDilatedConvolution()

                mv::Shape storageElementShape = mv::Shape({{inputTensorShape[mv::IO_WIDTH_DIMENSION]},
                                                           {inputTensorShape[mv::IO_HEIGHT_DIMENSION]},
                                                           {1},
                                                           {1}});
                std::vector<int64_t> storageElementData(storageElementShape.totalSize(), 0);
                std::string storageElementName = dpuTask->getName() + "storage_element_map";
                auto storageElement = om.constantInt(storageElementData, storageElementShape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, storageElementName);
                storageElement->set<bool>("dilatedSubConvSE", true);
                om.getSourceOp(storageElement)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                unsigned newSize = dpuTask->addInputTensor(storageElement);
                om.defineFlow(storageElement, dpuTask, newSize - 1);
                auto seTensorIdx = dpuTask->hasAttr("storageElementIndex") ?
                    dpuTask->get<std::vector<size_t>>("storageElementIndex") :
                    std::vector<size_t>();
                seTensorIdx.push_back(newSize - 1);
                dpuTask->set<std::vector<size_t>>("storageElementIndex", seTensorIdx);
            }
        }
    }
}

bool compilerSolvesSparsity(mv::Data::FlowListIterator flow)
{
    auto isDilatedConvRelated =
        flow.sink()->hasAttr("activationSparsityCompilerSolvingForDilatedConv") &&
        flow.sink()->get<bool>("activationSparsityCompilerSolvingForDilatedConv");

    if((!flow->getTensor()->isPopulated() &&
        flow.sink()->hasAttr("activationSparsityCompilerSolving") &&
        flow.sink()->get<bool>("activationSparsityCompilerSolving")) ||
        isDilatedConvRelated)
           return true;
    return false;
}

bool checkActivationSparsitySourceOpConditions(mv::Data::FlowListIterator flow)
{
    auto source = flow.source();
    if(!compilerSolvesSparsity(flow) &&
        source->getOpType() == "DPUTask" &&
        source->hasAttr("outputActivationSparsity") &&
        source->get<bool>("outputActivationSparsity"))
        return true;

    return false;
}

bool checkA0FloatSparsityBug(mv::Data::FlowListIterator flow, std::string referenceDevice)
{
    if (referenceDevice != "A0")
        return false;
    auto source = flow.source();
    auto sink = flow.sink();
    auto tensor = flow->getTensor();

    if(!tensor->isPopulated())
    {
        if((sink->hasAttr("floatPrecision") && sink->get<bool>("floatPrecision")) &&
                (source->hasAttr("mixedToFloat") && source->get<bool>("mixedToFloat")))
           return true;
        if((source->hasAttr("floatPrecision") && source->get<bool>("floatPrecision")) &&
                (sink->hasAttr("floatPrecision") && sink->get<bool>("floatPrecision")))
           return true;
    }
    return false;
}

// In VPU2 only sparse consumers are ZMajorConv and Eltwise
// Activation tensor sparsity can be solved:
// a) either by runtime with proper sparsity generation
// for ther below cases:
//      i) It is an output of a DPUTask.
//          (ODU populates storage element and sparsity map).
//      ii) Limitation in handling sparisty arise for the parent sparse out op cases:
//          SplitOverK, StreamOverK, workload KTiling
// b) or by compiler generated dummy sparse data (all 1's sparse map)
// in which case we no longer have the restrictions for the runtime generated sparsity
// (parent task is dpuTask and not K segmentable)

// An activation tensor MUST be sparse if it's:
// 1) SplitOverH ZMajorConvolution with kernel > 1 (A0 HW bug)
// 2) Float ZMajorConvolution and Eltwise DPU task (A0 HW bug)

static void setSparsityAttrForUnpopulatedFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::DataModel dm(model);
    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    auto referenceDevice = globalParams->get<std::string>("referenceDevice");

    for(auto tensor = dm.tensorBegin(); tensor != dm.tensorEnd(); ++tensor)
    {
        // Probably an inner tensor, skip
        if(!tensor->hasAttr("flows"))
            continue;

        // Populated tensor, skip
        if(tensor->isPopulated())
            continue;

        auto flows = tensor->get<std::set<std::string>>("flows");

        bool tensorSparsifiable = true;
        bool tensorNeedsSparsity = false;
        bool inputActivationSparsity = false;
        bool outputActivationSparsity = false;

        // Conditions for activation sparsity
        // 1) The source of the activation tensor must be a DPUTask (Otherwise there's no ODU to write SM, SE)
        // 2) The sink of the activation tensor must be a ZMajor Convolution, the only operation
        //    with an IDU capable of handling sparsity data
        // 3) Runtime doesn't support SOK and activation sparsity neither in input nor output
        // 4) HACK-Configuration...idu will have 1 case that the compiler generates sparsity info
        for(auto& flowStr: flows)
        {
            auto flow = dm.getDataFlow(flowStr);
            if (flow.sink()->isSparsityConsumer() &&
                (checkA0SOHSparsityBug(flow, referenceDevice) ||
                checkA0FloatSparsityBug(flow, referenceDevice)) &&
                !compilerSolvesSparsity(flow))
            {
                tensorNeedsSparsity = true;
                break;
            }
        }
        for(auto& flowStr: flows)
        {
            auto flow = dm.getDataFlow(flowStr);
            auto source = flow.source();
            auto sink = flow.sink();
            outputActivationSparsity |= source->hasAttr("outputActivationSparsity") ? source->get<bool>("outputActivationSparsity") : false;
            inputActivationSparsity |= sink->hasAttr("inputActivationSparsity") ? sink->get<bool>("inputActivationSparsity") : false;
            if(source->getOpType() != "DPUTask" ||
               source->get<std::string>("splitStrategy") == "SplitOverK" ||
               sink->getOpType() != "DPUTask" ||
               !sink->isSparsityConsumer())
            {
                tensorSparsifiable = false;
                break;
            }
        }

        if(tensorNeedsSparsity && !tensorSparsifiable)
            throw std::runtime_error("Wrong strategy generated: tensor " + tensor->getName() + " needs sparsity but it can't be sparsified");
        if((tensorSparsifiable && inputActivationSparsity && outputActivationSparsity) || tensorNeedsSparsity)
        {
            tensor->set<bool>("needs_sparse", true);

            //Now that we know tensor will be sparse, mark the input as need subtensor aligned
            auto sourceOp = om.getSourceOp(tensor);

            if (sourceOp->getOpType() == "DPUTask" && (sourceOp->get<std::string>("taskOp") == "Conv" || sourceOp->get<std::string>("taskOp") == "DepthwiseConv"))//TODO More??
            {
                sourceOp->getInputTensor()[0]->set<bool>("needs_splits_aligned", true);
                // Handle Align op's input tensor split-alignment
                auto parentOp = om.getSourceOp(sourceOp->getInputTensor()[0]);
                if (parentOp->getOpType() == "Align")
                    parentOp->getInputTensor()[0]->set<bool>("needs_splits_aligned", true);
            }
        }

    }
}
static void generateSparsityMapsUnpopulatedTensorsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::DataModel dm(model);

    for(auto tensor = dm.tensorBegin(); tensor != dm.tensorEnd(); ++tensor)
    {
        // Probably an inner tensor, skip
        if(!tensor->hasAttr("flows"))
            continue;

        // Populated tensor, skip
        if(tensor->isPopulated())
            continue;

        if((tensor->hasAttr("needs_sparse") && tensor->get<bool>("needs_sparse")))
        {
            tensor->setSparse();
        }
    }
}

static void generateSparsityMapsEltwiseFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    mv::OpModel om(model);
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType = opIt->getOpType();
        if (opType == "DPUTask")
        {
            std::string taskOp = opIt->get<std::string>("taskOp");
            if(taskOp == "Eltwise")
            {
                bool inputActivationSparsity = opIt->hasAttr("inputActivationSparsity") ? opIt->get<bool>("inputActivationSparsity") : false;

                if(!inputActivationSparsity)
                    continue;

                auto input0 = opIt->getInputTensor(0);
                auto input1 = opIt->getInputTensor(1);

                // Sparsity evaluation is dependant only on flow source and not on sink
                // In cases of ops branching and producing sparisty, either input of eltwise
                // will have the same source for all it's flows
                auto flow0 = om.getDataFlow(*input0->get<std::set<std::string>>("flows").begin());
                auto flow1 = om.getDataFlow(*input1->get<std::set<std::string>>("flows").begin());

                if(checkActivationSparsitySourceOpConditions(flow0) && checkActivationSparsitySourceOpConditions(flow1))
                {
                    input0->setSparse();
                    input1->setSparse();
                    // Note: odu_offset to be set on the input of the eltwise that results in positive number
                    // Store ref to tensor odu_offset will be calculated from, so we can find address at serialization
                    auto input0_op = om.getSourceOp(input0);
                    input0_op->set<std::string>("needsODUoffset", input1->getName());

                    auto input1_op = om.getSourceOp(input1);
                    input1_op->set<std::string>("needsODUoffset", input0->getName());
                }
            }
        }
    }
}

