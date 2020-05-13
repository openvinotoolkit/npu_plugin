#include "math.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/tensor/tiling.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry&,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void streamCopyOperationsFcn(const mv::pass::PassEntry&,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StreamingOperations)
        .setFunc(streamingOperationsFcn)
        .setDescription(
                "Generates New Ops according to Streaming Strategies that the graph provides");

        MV_REGISTER_PASS(StreamBinaryDataWeights)
        .setFunc(streamBinaryDataWeightsFcn)
        .setDescription(
            "The StreamOverK on Costant Operastions creates Constant + Slice, which is new smaller/fused Constants"
        );

        MV_REGISTER_PASS(StreamCopyOperations)
        .setFunc(streamCopyOperationsFcn)
        .setDescription(
            "This pass will handle the copy+slice pattern"
        );
    }
}

mv::Data::OpListIterator operationsReplacement(mv::Data::OpListIterator parentOpIt,
        mv::Data::TensorIterator sourceTensor,
        mv::OpModel om,
        mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    while(opIt.parentsSize() > 1)
    {
        auto paramOp = opIt.leftmostParent();
        ++paramOp;
        om.removeOp(paramOp);
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        //no need to trigger a cascade, we know what we are doing
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

struct opStreamingSplitDef
{
    std::string axis ;
    size_t numSplits ;
};

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling);
mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling);
mv::Data::TensorIterator solveBatchTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling);

std::map<std::string, std::function<mv::Data::TensorIterator(mv::ComputationModel&, mv::Data::OpListIterator, mv::Tiling&)>>
streamSplit =
{
    {"W",solveSpatialTiling},
    {"H",solveSpatialTiling},
    {"K",solveWeightsTiling},
    {"C",solveWeightsTiling}, //NOTE::Only Convolution/Depthwise is supported for SoK now
    {"N",solveBatchTiling}
};

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model,
        mv::Data::OpListIterator op,
        mv::Tiling& tiling)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto kernelTensor = op->getInputTensor(1);
    auto outputTensor = op->getOutputTensor(0);
    mv::Shape kernelShape  = kernelTensor->getShape();

    mv::QuantizationParams inputQuantParams = {{},{},{},{}};
    if(inputTensor->hasAttr("quantParams"))
        inputQuantParams = inputTensor->get<mv::QuantizationParams>("quantParams");

    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    auto attrsToCopy = op->getAttrs({"stride", "padding", "shape", "bias"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");

    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> newTensors(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    size_t biasStartIndex = 0;
    size_t biasEndIndex = 0;

    //todo::find a better location for this. Should not be slice.. but something like Copy layer... will do with dummy slice for speed
    //aslo.. have no idea why it's not working for the scenarion stream->concat->copySlice->stream when all is in CMX ... need debug.
    auto copyInput = om.slice(inputTensor,
                                mv::Shape({0,0,0,0}),
                                inputTensor->getShape(),
                                inputQuantParams,
                                inputTensor->getName() + op->getName() + "_KStreamCopyIn_");
    auto copyInputOp = om.getSourceOp(copyInput);
    copyInputOp->set<unsigned>("opId", opId);
    copyInputOp->set<std::string>("splitStrategy", splitStrategy);

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator slice;
        auto kernelSliceShape = childTiles[split].getKernelShape();
        auto kernelSliceStart = childTiles[split].getKernelStart();
        kernelSliceShape[mv::KERNEL_HEIGHT] = kernelShape[mv::KERNEL_HEIGHT]; //the tiling does not contain KERNEL W/H Info
        kernelSliceShape[mv::KERNEL_WIDTH] = kernelShape[mv::KERNEL_WIDTH];
        //todo:: clean this if-then-else quantParams logic
        if (kernelTensor->hasAttr("quantParams"))
        {
            auto sliceQuantParams = kernelTensor->get<mv::QuantizationParams>("quantParams");
            if (kernelTensor->get<mv::QuantizationParams>("quantParams").getScale().size() > 1)
            {
                std::size_t outputChannelsofSlice = 0, starting_point = 0;
                if (op->getOpType() == "Conv")
                {
                    outputChannelsofSlice = childTiles[split].getSize()[mv::KERNEL_OUTPUT_CHANNELS];
                    starting_point = childTiles[split].getStartCoord()[mv::KERNEL_OUTPUT_CHANNELS];
                }
                else if (op->getOpType() == "DepthwiseConv")
                {
                    outputChannelsofSlice = childTiles[split].getSize()[mv::KERNEL_INPUT_CHANNELS];
                    starting_point = childTiles[split].getStartCoord()[mv::KERNEL_INPUT_CHANNELS];
                }
                std::vector<double> scales(outputChannelsofSlice);
                std::vector<int64_t> zeros(outputChannelsofSlice);
                for (std::size_t i = starting_point; i < starting_point + outputChannelsofSlice; i++)
                {
                    scales.at(i - starting_point) = sliceQuantParams.getScale()[i];
                    zeros.at(i - starting_point) = sliceQuantParams.getZeroPoint()[i];
                }
                sliceQuantParams = mv::QuantizationParams(zeros,
                                                            scales,
                                                            sliceQuantParams.getMin(),
                                                            sliceQuantParams.getMax());
            }

            slice = om.slice(kernelTensor,
                            kernelSliceStart,
                            kernelSliceShape,
                            sliceQuantParams,
                            kernelTensor->getName() + inputTensor->getName() + "_sliceK" + std::to_string(split));
        }
        else
        {
            slice = om.slice(kernelTensor,
                            kernelSliceStart,
                            kernelSliceShape,
                            {{}, {}, {}, {}},
                            kernelTensor->getName() + "_sliceK" + std::to_string(split));
        }
        om.getSourceOp(slice)->set<unsigned>("opId", opId);

        std::string streamingOpName = op->getName() + "_streamK" + std::to_string(split);
        mv::Data::TensorIterator conv;
        //todo:: clean this if-then-else conv/DpthwiseConv logic... it's just bloatware code

        if (op->getOpType() == "Conv")
        {
            //todo:: place it in a more generic location

            conv = om.conv(copyInput,
                                    slice,
                                    op->get("stride"),
                                    op->get("padding"),
                                    op->get<unsigned>("dilationFactor"),
                                    op->get<unsigned>("group"),
                                    op->get<mv::DType>("dType"),
                                    op->get<mv::QuantizationParams>("quantParams"),
                                    streamingOpName);
        }
        else if (op->getOpType() == "DepthwiseConv")
        {
            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            mv::Data::TensorIterator sliceInput = om.slice(copyInput,
                                sliceStart,
                                sliceShape,
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_sliceHK_" + std::to_string(split));

            conv = om.depthwiseConv(sliceInput,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);
            auto sliceInputOp = om.getSourceOp(sliceInput);
            sliceInputOp->set<unsigned>("opId", opId);
            sliceInputOp->set<std::string>("splitStrategy", splitStrategy);
        }
        om.getSourceOp(conv)->set<unsigned>("opId", opId);

        //todo: clean this if-then-else bias logic.... bloatware code....
        if (op->hasAttr("bias"))
        {
            auto tileSize = kernelSliceShape[axisToSplit];
            biasStartIndex = kernelSliceStart[axisToSplit];
            biasEndIndex = biasStartIndex + tileSize;

            auto biasTensorName = op->get<std::string>("bias");
            auto originalBiasTensor = dm.getTensor(biasTensorName);
            auto oiginalBiasData = originalBiasTensor->getData();

            if ( biasEndIndex > oiginalBiasData.size())
                biasEndIndex = oiginalBiasData.size();
            std::vector<mv::DataElement>::const_iterator biasFirst = oiginalBiasData.begin() + biasStartIndex;
            std::vector<mv::DataElement>::const_iterator biasLast = oiginalBiasData.begin() + biasEndIndex;
            std::vector<mv::DataElement> subBiasData(biasFirst, biasLast);
            std::string newBiasTensorName = mv::createBiasName(op->getName() + "_split_" + std::to_string(split));
            mv::Data::TensorIterator biasTensor;
            mv::Data::TensorIterator biasTensorX;
            if (originalBiasTensor->hasAttr("quantParams"))
            {
                auto biasAttrQPs = originalBiasTensor->get("quantParams");
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData, biasAttrQPs ));
            }
            else
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData));
            om.addAttr(om.getSourceOp(conv), "bias", biasTensorX->getName());
        }
        auto newOp = om.getSourceOp(conv);

        newOp->set<bool>("splitted",true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->setAttrs(attrsToCopy);

        slices[split] = slice;
        newTensors[split] = conv;

        bool enableSerialStreaming = true;
        if ((split>0)&&(enableSerialStreaming))
            cm.defineFlow(om.getSourceOp(newTensors[split-1]), om.getSourceOp(newTensors[split]));
    }

    kernelTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    //in case of non-symmetric stream, we neet to check if at least one op is the last in the chain
    bool atLeastOneOpIsLast = false;
    for (unsigned idx = 0 ; idx < number_of_splits ; ++idx)
    {
        auto slice = slices[idx];
        auto newTensor = newTensors[idx];
        mv::Tensor::MemoryLocation inputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        mv::Tensor::MemoryLocation outputLocation(mv::Tensor::MemoryLocation::DEFAULT);

        auto numChildStreames = tiling.childTiles()[idx].childTiles().size();

        if(numChildStreames > 1)
        {
            //todo::should not be this convoluted to get the parentTensor of a tensor .....
            //layer may have multiple inputs with different locations (eltwise). Each inputTensor will get a slice layer based on the stream
            //so, for deciding the location of the slice, we have to check each input of the slice respectively
            inputLocation.relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            atLeastOneOpIsLast = true;
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        }
        slice->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
        newTensor->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
    }
    //todo::better solution for this... need to decide on the location of the CopyInput
    {
        if(atLeastOneOpIsLast)
            copyInput->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);
        else
            copyInput->set<mv::Tensor::MemoryLocation>("Location",inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
    }

    for(unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if(childTiles[split].childTiles().size() > 1)
        {
            auto newStreamAxis = childTiles[split].getAxis();
            auto newStreamFunc = streamSplit[newStreamAxis];

            out = newStreamFunc(om,om.getSourceOp(newTensors[split]),childTiles[split]);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
        {
            out = newTensors[split];
        }
        final_outputs[split] = out;
    }

    auto concat = om.concat(final_outputs,
                    "C",
                    op->get<mv::DType>("dType"),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);

    concat->set<mv::Tensor::MemoryLocation>("Location",outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model,
                    mv::Data::OpListIterator op,
                    mv::Tiling& tiling)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto outputTensor = op->getOutputTensor("output");
    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    // NOTE: In the streaming case, we can't just blindly copy everything like we
    // do in the DPUTask conversion case. We have to overwrite shape, padding, etc.
    auto attrsToCopy = op->getAttrs({"stride", "padding", "shape"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");

    std::vector<mv::Shape> spatial_indexes(number_of_splits);
    std::vector<mv::Data::TensorIterator> slices;
    std::vector<mv::Data::TensorIterator> newTensors(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    std::array<unsigned short, 2> kernelStride;
    if (op->hasAttr("stride"))
        kernelStride = op->get<std::array<unsigned short, 2>>("stride");
    else
        kernelStride = {1,1};//fake stride

    //NOTE: assuming order of paddings: left,right,top,bottom
    std::array<unsigned short, 4> padding;
    if (op->hasAttr("padding"))
        padding = op->get<std::array<unsigned short, 4>>("padding");
    else
        padding = {0, 0, 0, 0};

    auto startPad = padding;
    auto endPad = padding;
    auto middlePad = padding;
    auto currentPad = padding;

    if (axisToSplit == mv::Shape::getAxis("W"))
    {
        startPad[1] = 0;
        endPad[0] = 0;
        middlePad[0] = 0;
        middlePad[1] = 0;
    }
    if (axisToSplit == mv::Shape::getAxis("H"))
    {
        startPad[3] = 0;
        endPad[2] = 0;
        middlePad[2] = 0;
        middlePad[3] = 0;
    }

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        if (split == 0)
            currentPad = startPad;
        else if (split == (number_of_splits -1))
            currentPad = endPad;
        else
            currentPad = middlePad;

        mv::Data::TensorIterator newTensor;
        std::string opType = op->getOpType();
        std::string streamingOpName = op->getName() + "_streamH" + std::to_string(split);
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv")
        {
            auto inputTensor = op->getInputTensor(0);

            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            mv::Data::TensorIterator slice = om.slice(inputTensor,
                                sliceStart,
                                sliceShape,
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_sliceH" + std::to_string(split));
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if (opType == "MaxPool")
                newTensor = om.maxPool(slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                currentPad,
                                op->get<const bool>("exclude_pad"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                1, //this is already taken care of, dont want to change kernel size again
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);

            if (opType == "Conv")
                newTensor = om.conv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                1, //this is already taken care of, dont want to change kernel size again
                                op->get<unsigned>("group"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);
            slices.push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            std::vector<mv::Data::TensorIterator> eltwiseSlices;
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->get<mv::DType>("dType");
            for (unsigned i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);

                auto sliceShape = childTiles[split].getActivationShape();
                auto sliceStart = childTiles[split].getActivationStart();

                auto slice = om.slice(inputTensor,
                                sliceStart,
                                sliceShape,
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_sliceH" + std::to_string(split) + "_" + std::to_string(i));
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices.push_back(slice);
                eltwiseSlices.push_back(slice);
            }

            newTensor = om.eltwise(eltwiseSlices,
                                    eltwiseType,
                                    originalDType,
                                    op->get<mv::QuantizationParams>("quantParams"),
                                    op->getName() + "_streamH" + std::to_string(split));
        }

        auto newOp = om.getSourceOp(newTensor);

        newOp->setAttrs(attrsToCopy);
        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated

        newTensors[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(newTensors[split-1]), om.getSourceOp(newTensors[split]));
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;
    // Unlike Kstream, we may have different number of "newOutputs" than slices, since Ops can have multiple inputs
    // and the concept of stream is per OP not per tensor // todo:: find way with less duplication of code&logic

    auto numChildStreames = tiling.childTiles().size();
    for (auto newTensor : newTensors)
    {
        mv::Tensor::MemoryLocation outputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        else
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        newTensor->set<mv::Tensor::MemoryLocation>("Location",outputLocation);

    }
    for (auto slice : slices)
    {
        mv::Tensor::MemoryLocation inputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
        {
            auto sliceInputTensor = om.getSourceOp(slice)->getInputTensor(0);
            inputLocation .relocate(sliceInputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        }
        slice->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            auto newStreamAxis = childTiles[split].getAxis();
            auto newStreamFunc = streamSplit[newStreamAxis];

            out = newStreamFunc(om, om.getSourceOp(newTensors[split]), childTiles[split]);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
            out = newTensors[split];
        final_outputs[split] = out;
    }

    auto concat = om.concat(final_outputs,
                    tiling.getAxis(),
                    op->get<mv::DType>("dType"),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

mv::Data::TensorIterator solveBatchTiling(mv::ComputationModel& model,
                    mv::Data::OpListIterator op,
                    mv::Tiling& tiling)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto outputTensor = op->getOutputTensor("output");
    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto childTiles = tiling.childTiles();

    // NOTE: In the streaming case, we can't just blindly copy everything like we
    // do in the DPUTask conversion case. We have to overwrite shape, padding, etc.
    auto attrsToCopy = op->getAttrs({"stride", "padding", "shape"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");

    std::vector<mv::Data::TensorIterator> slices;
    std::vector<mv::Data::TensorIterator> newTensors(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    std::array<unsigned short, 2> kernelStride;
    if (op->hasAttr("stride"))
        kernelStride = op->get<std::array<unsigned short, 2>>("stride");
    else
        kernelStride = {1,1};//fake stride

    //NOTE: assuming order of paddings: left,right,top,bottom
    std::array<unsigned short, 4> padding;
    if (op->hasAttr("padding"))
        padding = op->get<std::array<unsigned short, 4>>("padding");
    else
        padding = {0, 0, 0, 0};

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator newTensor;
        std::string opType = op->getOpType();
        std::string streamingOpName = op->getName() + "_stream" + tiling.getAxis() + std::to_string(split);
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv")
        {
            auto inputTensor = op->getInputTensor(0);

            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            mv::Data::TensorIterator slice = om.slice(inputTensor,
                                sliceStart,
                                sliceShape,
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_slice" + tiling.getAxis() + std::to_string(split));
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if (opType == "MaxPool")
                newTensor = om.maxPool(slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                padding,
                                op->get<const bool>("exclude_pad"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                padding,
                                1, //this is already taken care of, dont want to change kernel size again
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);

            if (opType == "Conv")
                newTensor = om.conv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                padding,
                                1, //this is already taken care of, dont want to change kernel size again
                                op->get<unsigned>("group"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);
            slices.push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->get<mv::DType>("dType");
            for (unsigned i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);

                auto sliceShape = childTiles[split].getActivationShape();
                auto sliceStart = childTiles[split].getActivationStart();

                auto slice = om.slice(inputTensor,
                                sliceStart,
                                sliceShape,
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_slice"  + tiling.getAxis() + std::to_string(split) + "_" + std::to_string(i));
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices.push_back(slice);
            }

            newTensor = om.eltwise(slices,
                                    eltwiseType,
                                    originalDType,
                                    op->get<mv::QuantizationParams>("quantParams"),
                                    op->getName() + "_stream" + tiling.getAxis() + std::to_string(split));
        }

        auto newOp = om.getSourceOp(newTensor);

        newOp->setAttrs(attrsToCopy);
        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated

        newTensors[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(newTensors[split-1]), om.getSourceOp(newTensors[split]));
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;
    // Unlike Kstream, we may have different number of "newOutputs" than slices, since Ops can have multiple inputs
    // and the concept of stream is per OP not per tensor // todo:: find way with less duplication of code&logic

    auto numChildStreames = tiling.childTiles().size();
    for (auto newTensor : newTensors)
    {
        mv::Tensor::MemoryLocation outputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        else
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        newTensor->set<mv::Tensor::MemoryLocation>("Location",outputLocation);

    }
    for (auto slice : slices)
    {
        mv::Tensor::MemoryLocation inputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
        {
            auto sliceInputTensor = om.getSourceOp(slice)->getInputTensor(0);
            inputLocation .relocate(sliceInputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        }
        slice->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            auto newStreamAxis = childTiles[split].getAxis();
            auto newStreamFunc = streamSplit[newStreamAxis];

            out = newStreamFunc(om, om.getSourceOp(newTensors[split]), childTiles[split]);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
            out = newTensors[split];
        final_outputs[split] = out;
    }

    auto concat = om.concat(final_outputs,
                    tiling.getAxis(),
                    op->get<mv::DType>("dType"),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element&,
                                mv::Element&)
{

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    //NOTE: NESTED STREAMING MEANS 2 LEVELS OF STREAMING, eg. HK, Stream Over H will stream
    //the input Tensor of the Op and then for every new Op have to stream it over K, which
    //means the weights will be repeated for the second level of streaming, this is why need
    //the data structures below...to create only one pair of nested slices

    for (auto layerNameStrategy : strategyList)
    {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        //NOTE: Graph optimizer will never do that but needs to be here for manual Scheduling
        if (!om.checkOp(nodeName))
        {
            pass.log(mv::Logger::MessageType::Debug, nodeName + " is not present in model, skipping streaming");
            continue;
        }
        auto opIt =  om.getOp(nodeName);
        std::string opType = opIt->getOpType();

        //For now do streaming pass only for the DPU layers
        if ((opType != "Conv") && (opType != "DepthwiseConv") && (opType != "MaxPool") && (opType != "Eltwise"))
            continue;

        auto outputTensor = opIt->getOutputTensor(0);
        auto outputShape = outputTensor->getShape();
        auto inputTensor = opIt->getInputTensor(0);
        auto inputShape = inputTensor->getShape();

        auto zeroStartAxis = mv::Shape({0,0,0,0});
        mv::Tiling masterTile;
        if((opType == "Conv") || (opType == "DepthwiseConv"))
        {
            //op has kernel
            auto kernelShape = opIt->getInputTensor(1)->getShape();
            masterTile = mv::Tiling(inputShape,kernelShape);
        }
        else
        {
            //for multi-input ops, this pass is assuming that all inputs are equalt, and the streams happens simetrically (Eltwise)
            masterTile = mv::Tiling(inputShape);
        }

        auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");

        std::vector<mv::Tiling*> tiles = {&masterTile};

        auto applyTiling = [&](mv::Element& split, mv::Tiling& tile) -> std::vector<mv::Tiling>*
        {
            //the axis&split are stored in a map with key-> val .....
            //Don't want to if-then-else over all possible values of the axis...
            //the map should have only one key.. this is the draw-back of too generic mv::Element
            auto axis = split.attrsKeys()[0];
            auto numSplits = split.get<int>(axis);

            pass.log(mv::Logger::MessageType::Debug, opIt->getName() +
                " " + axis + " : " + std::to_string(numSplits));
            if(numSplits > 1)
            {
                tile.setAxis(axis);
                tile.resizeNumberOfTiles(numSplits);
                tile.generateTiling(opIt);
                return &tile.childTiles();
            }
            else
            {
                return nullptr;
            }
        };

        for (auto split : splitList)
        {
            std::vector<mv::Tiling*> newChildTiles(0);
            for(auto tile : tiles)
            {
                auto childTiles = applyTiling(split,*tile);
                if(childTiles)
                {
                    for(auto& childTile : *childTiles)
                    {
                        newChildTiles.push_back(&childTile);
                    }
                }
                else
                {
                    newChildTiles.push_back(tile);
                }
            }
            tiles = newChildTiles;
        }

        if(masterTile.childTiles().size() > 1)
        {
            auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile);

            // reconnect children to subgraph
            std::vector<mv::Data::OpListIterator> opsToLink;
            std::vector<std::size_t> inputSlots;
            for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                opsToLink.push_back(sinkFlow.sink());
                inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            }

            om.removeOp(opIt);
            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                opsToLink[j]->setInputTensor(result, inputSlots[j], false);
                om.defineFlow(result, opsToLink[j], inputSlots[j]);
            }
        }
}
}


static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry& ,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& ,
                                        mv::Element& ,
                                        mv::Element &)
{
    //Need to duplicate the consts to number equal to streams, cause of the binary_data
    mv::OpModel om(model);

    std::set <std::string> removeConstantsSet;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();
        std::vector<mv::Data::TensorIterator> toSort;

        if (opType == "Slice" && opIterator->getInputTensor(0)->isPopulated())
        {
            auto inTensorSlice = opIterator->getInputTensor(0);
            removeConstantsSet.insert(om.getSourceOp(inTensorSlice)->getName());
            auto outTensorSlice = opIterator->getOutputTensor(0);
            auto parentOpIt = om.getSourceOp(opIterator->getInputTensor(0));
            mv::QuantizationParams tensorQuantizationParams = {{},{},{},{}};
            auto shape = outTensorSlice->getShape();
            if (outTensorSlice->isQuantized())
                tensorQuantizationParams = outTensorSlice->get<mv::QuantizationParams>("quantParams");

            auto newConstant = om.constantDataElement(outTensorSlice->getData(), shape,
                                                               outTensorSlice->getDType(), outTensorSlice->getOrder(),
                                                               tensorQuantizationParams, opIterator->getName() + "_weights");
            newConstant->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
            auto constantOp = om.getSourceOp(newConstant);
            if(opIterator->hasAttr("opId"))
            {
                unsigned currentOpId = opIterator->get<unsigned>("opId");
                constantOp->set<unsigned>("opId", currentOpId);
            }
            opIterator = operationsReplacement(parentOpIt, newConstant, om, opIterator);
        }
    }
    for (auto& opName:removeConstantsSet)
        om.removeOp(om.getOp(opName));
}

static void streamCopyOperationsFcn(const mv::pass::PassEntry& ,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& ,
                                        mv::Element& ,
                                        mv::Element &)
{
    //Need to duplicate the consts to number equal to streams, cause of the binary_data
    mv::OpModel om(model);

    std::set <std::string> removeCopySet;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();

        if (opType == "Slice" && (!opIterator->getInputTensor(0)->isPopulated()))
        {
            auto previousOp = om.getSourceOp(opIterator->getInputTensor(0));
            if (previousOp->getOpType() == "Copy")
            {
                opIterator->setInputTensor(previousOp->getInputTensor(0), 0, false);
                om.defineFlow(previousOp->getInputTensor(0),opIterator , 0);
                removeCopySet.insert(previousOp->getName());
            }
        }
    }
    for (auto& opName:removeCopySet)
        om.removeOp(om.getOp(opName));
}
