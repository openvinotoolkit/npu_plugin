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
        mv::OpModel & om,
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

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, bool vertical_fusion_overlap);
mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, bool vertical_fusion_overlap);
mv::Data::TensorIterator solveBatchTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, bool vertical_fusion_overlap);

std::map<std::string, std::function<mv::Data::TensorIterator(mv::ComputationModel&, mv::Data::OpListIterator, mv::Tiling&, bool vertical_fusion_overlap)>>
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
        mv::Tiling& tiling, bool /*vertical_fusion_overlap*/ = false)
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
    auto kernelOp = om.getSourceOp(kernelTensor);

    auto inputQuantParams  = inputTensor->getQuantParams();
    auto outputQuantParams = outputTensor->getQuantParams();

    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    // Attributes query based on blacklist
    // Weights K || C (depthwise ops) stream, need only overwrite shape and bias
    auto attrsToCopy = op->getAttrs({"shape", "bias"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    bool mixedToFloat = false;

    if(op->hasAttr("mixedToFloat"))
        mixedToFloat = op->get<bool>("mixedToFloat");

    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> newTensors(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    size_t biasStartIndex = 0;
    size_t biasEndIndex = 0;

    bool isDilatedConv = op->hasAttr("DilatedSubConv") && op->get<bool>("DilatedSubConv");
    bool avoidCmxConcat = op->hasAttr("avoidCmxConcat") && op->get<bool>("avoidCmxConcat");

    //todo::find a better location for this. Should not be slice.. but something like Copy layer... will do with dummy slice for speed
    //aslo.. have no idea why it's not working for the scenarion stream->concat->copySlice->stream when all is in CMX ... need debug.
    mv::Data::TensorIterator copyInput;
    if(om.getSourceOp(inputTensor)->getOpType() != "Slice")
    {
        copyInput = om.slice(inputTensor->getName() + op->getName() + "_KStreamCopyIn_",
                             inputTensor,
                             mv::Shape({0,0,0,0}),
                             inputTensor->getShape());
        copyInput->setQuantParams(inputQuantParams);
        auto copyInputOp = om.getSourceOp(copyInput);
        copyInputOp->set<unsigned>("opId", opId);
        copyInputOp->set<std::string>("splitStrategy", splitStrategy);
    }
    else
    {
        copyInput = inputTensor;
    }

    //NOTE: the idea here is that the n-1 first splits will be symmetrical on h/w
    //so in order to concatenate later for the dilation case we will need to know
    //the dim of the n-1 first streams and this should be stored in the last stream
    std::size_t symmetrical_first_dimension = 0;
    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator slice;
        auto kernelSliceShape = childTiles[split].getKernelShape();
        auto kernelSliceStart = childTiles[split].getKernelStart();
        kernelSliceShape[mv::KERNEL_HEIGHT] = kernelShape[mv::KERNEL_HEIGHT]; //the tiling does not contain KERNEL W/H Info
        kernelSliceShape[mv::KERNEL_WIDTH] = kernelShape[mv::KERNEL_WIDTH];

        if (isDilatedConv &&
                kernelOp->hasAttr("dilationConvKernelSliced")
                && kernelOp->get<bool>("dilationConvKernelSliced")) //already handled this dilated Conv, nothing to do
        {
            //find the proper slice
            bool sliceFound = false;
            for (auto sinkFlow = kernelOp.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                auto sinkOp = sinkFlow.sink();
                if (sinkOp->getOpType() == "Slice" && sinkOp->hasAttr("dilatedConvKernelSliceIdx")
                        && sinkOp->get<unsigned>("dilatedConvKernelSliceIdx") == split)
                {
                    slice = sinkOp->getOutputTensor(0);
                    sliceFound = true;
                    break;
                }
            }
            if (!sliceFound)
                throw mv::RuntimeError("Streaming", "Slice for dilatedConv weights hasn't been found although kernel was marked as already Sliced!");
        }
        else
        {

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

                slice = om.slice(kernelTensor->getName() + inputTensor->getName() + "_sliceK" + std::to_string(split),
                                kernelTensor,
                                kernelSliceStart,
                                kernelSliceShape);
                slice->setQuantParams(sliceQuantParams);
            }
            else
            {
                slice = om.slice(kernelTensor->getName() + "_sliceK" + std::to_string(split),
                                kernelTensor,
                                kernelSliceStart,
                                kernelSliceShape);
            }
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if(isDilatedConv) //first time streaming if we are here, mark slice index for other subConvs
            {
                om.getSourceOp(slice)->set<unsigned>("dilatedConvKernelSliceIdx", split);
            }
        }
        std::string streamingOpName = op->getName() + "_streamK" + std::to_string(split);
        mv::Data::TensorIterator newTensor;
        //todo:: clean this if-then-else conv/DpthwiseConv logic... it's just bloatware code

        if (op->getOpType() == "Conv")
        {
            //todo:: place it in a more generic location

            newTensor = om.conv(streamingOpName,
                                copyInput,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"));
            newTensor->setQuantParams(outputQuantParams);
            newTensor->setDType(outputTensor->getDType());
            newTensor->setOrder(mv::Order("NHWC"));

            if (split != number_of_splits - 1)
                symmetrical_first_dimension = newTensor->getShape()[mv::IO_CHANNEL_DIMENSION];

            if ((op->hasAttr("DilatedSubConv") && op->get<bool>("DilatedSubConv")) || (op->hasAttr("DeconvSubConv") && op->get<bool>("DeconvSubConv")))
            {
                om.getSourceOp(newTensor)->set<unsigned>("streamKId", split);
                om.getSourceOp(newTensor)->set<std::size_t>("symmetrical_first_dimensionK",
                                                                symmetrical_first_dimension);
            }
        }
        else if (op->getOpType() == "DepthwiseConv")
        {
            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            auto sliceInput = om.slice(op->getName() + "_sliceHK_" + std::to_string(split),
                                copyInput,
                                sliceStart,
                                sliceShape);
            sliceInput->setQuantParams(inputQuantParams);

            newTensor = om.depthwiseConv(streamingOpName,
                                sliceInput,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"));
            newTensor->setQuantParams(outputQuantParams);
            if((op->hasAttr("asymmetricKernel")))
            {
                om.getSourceOp(newTensor)->set<unsigned>("asymmetricKernel", op->get<unsigned>("asymmetricKernel"));
            }
            auto sliceInputOp = om.getSourceOp(sliceInput);
            sliceInputOp->set<unsigned>("opId", opId);
            sliceInputOp->set<std::string>("splitStrategy", splitStrategy);
        }

        // Does more harm than good, since mixed precision is not treated correctly
        // further on
        // // Restore original out dtype, to account for mixed precision cases
        // // where we don't want the same datatype for output as the input tensors
        // newTensor->setDType(op->getOutputTensor(0)->getDType());
        om.getSourceOp(newTensor)->set<unsigned>("opId", opId);
        om.getSourceOp(newTensor)->set<std::string>("parentOpName", op->getName());

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
            mv::Data::TensorIterator biasTensorX;
            if (originalBiasTensor->hasAttr("quantParams"))
            {
                auto biasAttrQPs = originalBiasTensor->get("quantParams");
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData, biasAttrQPs ));
            }
            else
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData));
            om.addAttr(om.getSourceOp(newTensor), "bias", biasTensorX->getName());
        }
        auto newOp = om.getSourceOp(newTensor);

        newOp->set<bool>("splitted",true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->setAttrs(attrsToCopy);

        slices[split] = slice;
        newTensors[split] = newTensor;

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

            out = newStreamFunc(om,om.getSourceOp(newTensors[split]),childTiles[split], false);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
        {
            out = newTensors[split];
        }
        final_outputs[split] = out;
    }

    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    "C");
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(outputQuantParams);

    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    if(op->hasAttr("schedule_for_dpu_dma_overlap"))
    {
        auto pipelineId = op->get<unsigned>("schedule_for_dpu_dma_overlap");
        om.getSourceOp(concat)->set<unsigned>("schedule_for_dpu_dma_overlap", pipelineId);
    }
    if(avoidCmxConcat)
        om.getSourceOp(concat)->set<bool>("avoid_cmx_concat", true);

    if(mixedToFloat)
        om.getSourceOp(concat)->set<bool>("mixedToFloat", mixedToFloat);

    concat->set<mv::Tensor::MemoryLocation>("Location",outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
    if(isDilatedConv && !kernelOp->hasAttr("dilationConvKernelSliced"))
    {
        kernelOp->set<bool>("dilationConvKernelSliced", true);
    }
    return concat;
}

mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model,
                    mv::Data::OpListIterator op,
                    mv::Tiling& tiling, bool vertical_fusion_overlap = false)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    // Attributes query based on blacklist
    // Spatial H || W stream, need only overwrite shape, padding
    auto attrsToCopy = op->getAttrs({"padding", "shape"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    bool avoidCmxConcat = op->hasAttr("avoidCmxConcat") && op->get<bool>("avoidCmxConcat");
    bool concatTail = false;
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
    std::size_t symmetrical_first_dimension = 0;
    std::size_t symmetrical_first_dimension_input = 0;
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
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv" || opType == "HwConvert")
        {
            auto inputTensor  = op->getInputTensor(0);
            auto outputTensor = op->getOutputTensor(0);

            auto inputQuantParams  = inputTensor->getQuantParams();
            auto outputQuantParams = outputTensor->getQuantParams();

            auto outputDType = outputTensor->getDType();

            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            bool fusedConcatReshape = outputTensor->hasAttr("fusedConcatReshape") && outputTensor->get<bool>("fusedConcatReshape");

            auto slice = om.slice(op->getName() + "_sliceH" + std::to_string(split),
                                inputTensor,
                                sliceStart,
                                sliceShape);
            slice->setQuantParams(inputQuantParams);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if (opType == "MaxPool")
                newTensor = om.maxPool(streamingOpName,
                                slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                currentPad,
                                op->get<const bool>("exclude_pad"));

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"));

            if (opType == "Conv") {
                newTensor = om.conv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"));
                newTensor->setOrder(mv::Order("NHWC"));
            }

            if (opType == "HwConvert")
                newTensor = om.hwConvert(streamingOpName,
                                slice,
                                outputDType);

            newTensor->setDType(outputDType);
            newTensor->setQuantParams(outputQuantParams);
            if (fusedConcatReshape)
            {
                std::size_t numberOfConvsForAsymmetricalStride = outputTensor->hasAttr("numberOfConvsForAsymmetricalStride") ? outputTensor->get<std::size_t>("numberOfConvsForAsymmetricalStride") : 0;
                std::size_t asymmetricConvIndex = outputTensor->hasAttr("asymmetricConvIndex") ? outputTensor->get<std::size_t>("asymmetricConvIndex") : 0;
                newTensor->set<bool>("fusedConcatReshape", fusedConcatReshape);
                newTensor->set<std::size_t>("numberOfConvsForAsymmetricalStride", numberOfConvsForAsymmetricalStride);
                newTensor->set<std::size_t>("asymmetricConvIndex", asymmetricConvIndex);
            }

            if (split != number_of_splits - 1)
            {
                symmetrical_first_dimension = newTensor->getShape()[mv::IO_HEIGHT_DIMENSION];
            }
            if ((op->hasAttr("DilatedSubConv") && op->get<bool>("DilatedSubConv")) || (op->hasAttr("DeconvSubConv") && op->get<bool>("DeconvSubConv")))
            {
                om.getSourceOp(newTensor)->set<unsigned>("streamHId", split);
                om.getSourceOp(newTensor)->set<std::size_t>("symmetrical_first_dimensionH"
                                                         , symmetrical_first_dimension);
            }
            symmetrical_first_dimension_input += slice->getShape()[mv::IO_HEIGHT_DIMENSION];
            if((op->hasAttr("asymmetricKernel")))
                om.getSourceOp(newTensor)->set<unsigned>("asymmetricKernel", op->get<unsigned>("asymmetricKernel"));
            slices.push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            std::vector<mv::Data::TensorIterator> eltwiseSlices;
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->getOutputTensor(0)->getDType();
            for (unsigned i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                auto inputQuantParams = inputTensor->getQuantParams();

                auto sliceShape = childTiles[split].getActivationShape();
                auto sliceStart = childTiles[split].getActivationStart();

                auto slice = om.slice(op->getName() + "_sliceH" + std::to_string(split) + "_" + std::to_string(i),
                                inputTensor,
                                sliceStart,
                                sliceShape);
                slice->setQuantParams(inputQuantParams);
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices.push_back(slice);
                eltwiseSlices.push_back(slice);
            }

            auto quantParams = op->getOutputTensor(0)->getQuantParams();
            newTensor = om.eltwise(op->getName() + "_streamH" + std::to_string(split),
                                eltwiseSlices,
                                eltwiseType);
            newTensor->setDType(originalDType);
            newTensor->setQuantParams(quantParams);
        }

        // Restore original out dtype, to account for mixed precision cases
        // where we don't want the same datatype for output as the input tensors
        newTensor->setDType(op->getOutputTensor(0)->getDType());
        auto newOp = om.getSourceOp(newTensor);
        bool op_is_low_level_vf_input = op->hasAttr("inputLowLevel") && op->get<bool>("inputLowLevel");
        if (vertical_fusion_overlap)
        {
            newOp->set<std::size_t>("verticalFusionOutputOverlap", childTiles[split].getActivationStart()["H"]);
            newTensor->set<std::size_t>("verticalFusionOutputOverlap", childTiles[split].getActivationStart()["H"]);
            if (split != 0)
            {
                newOp->set<bool>("vertical_fusion_overlap", op_is_low_level_vf_input);
                auto previousLinesComputing = childTiles[split - 1].getActivationStart()["H"] +
                    childTiles[split - 1].getActivationShape()["H"];
                auto overLappingLines = previousLinesComputing - childTiles[split].getActivationStart()["H"];
                newOp->set<std::size_t>("concatOverLappingLines", overLappingLines);
            }
        }
        if (op_is_low_level_vf_input && (split != 0))
        {
            newOp->set<bool>("inputLowLevel", op_is_low_level_vf_input);
            auto previousLinesComputing = childTiles[split - 1].getActivationStart()["H"] +
                childTiles[split - 1].getActivationShape()["H"];
            auto overLappingLines = previousLinesComputing - childTiles[split].getActivationStart()["H"] - 1;
            if (op->hasAttr("overLappingSubgraphOpsIndex"))
            {
                //note: normally this constant needs to be computed and setted in vertical_fusion.cpp
                // in storeOverlappingEltwiseLines function, however since we target yolov4 should be ok now
                std::size_t overLappingLayersStrideKernel = 2;
                overLappingLines = overLappingLines - (overLappingLayersStrideKernel * op->get<std::size_t>("overLappingSubgraphOpsIndex"));
            }
            newOp->set<std::size_t>("doubleTensorOverlappingLines", overLappingLines);
        }
        om.getSourceOp(newTensor)->set<std::string>("parentOpName", op->getName());
        newOp->set<bool>("shareWeights", true);

        newOp->setAttrs(attrsToCopy);
        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated
        if (newOp->hasWeights())
            newOp->set<bool>("multiple_weight_out_degree", true);

        newTensors[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(newTensors[split-1]), om.getSourceOp(newTensors[split]));

        if (newOp->hasAttr("verticalFusionSubgraphTail") && newOp->get<bool>("verticalFusionSubgraphTail"))
            concatTail = true;
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;
    // Unlike Kstream, we may have different number of "newOutputs" than slices, since Ops can have multiple inputs
    // and the concept of stream is per OP not per tensor // todo:: find way with less duplication of code&logic

    auto outputTensor = op->getOutputTensor("output");
    auto numChildStreames = tiling.childTiles().size();
    for (auto newTensor : newTensors)
    {
        mv::Tensor::MemoryLocation outputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
        {
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            //NOTE: the idea here is that if you are not aligned with 16 channels in case that you are a z-maj
            //operation later you will have added the mechanism of align crop operation dmas to solve the //16
            //so if you are the last layer do not populate the output as a location but the ddr, leaving it in comments
            //as it is used mainly for the modelCutter, normally the locations should be handled in the placement of the
            //crop,align, quantize etc...
//            if ((newTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % 16 != 0) &&
//                    outputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::OUTPUT)
//                outputLocation.relocate(mv::Tensor::MemoryLocation::DDR);
        }
        else
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        newTensor->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
        if ((op->hasAttr("verticalFusionSubgraphHead") && op->get<bool>("verticalFusionSubgraphHead")) ||
            (op->hasAttr("verticalFusion") && op->get<bool>("verticalFusion")))
            newTensor->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);
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
        if ((op->hasAttr("verticalFusionSubgraphTail") && op->get<bool>("verticalFusionSubgraphTail")) ||
            (op->hasAttr("verticalFusion") && op->get<bool>("verticalFusion")))
            slice->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            auto newStreamAxis = childTiles[split].getAxis();
            auto newStreamFunc = streamSplit[newStreamAxis];

            out = newStreamFunc(om, om.getSourceOp(newTensors[split]), childTiles[split], false);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
            out = newTensors[split];
        final_outputs[split] = out;
    }

    auto quantParams = op->getOutputTensor(0)->getQuantParams();
    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    tiling.getAxis());
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(quantParams);
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    if (concatTail)
        om.getSourceOp(concat)->set<bool>("concatTail", concatTail);

    if(op->hasAttr("schedule_for_dpu_dma_overlap"))
    {
        auto pipelineId = op->get<unsigned>("schedule_for_dpu_dma_overlap");
        om.getSourceOp(concat)->set<unsigned>("schedule_for_dpu_dma_overlap", pipelineId);
    }
    if(avoidCmxConcat)
        om.getSourceOp(concat)->set<bool>("avoid_cmx_concat", true);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

mv::Data::TensorIterator solveBatchTiling(mv::ComputationModel& model,
                    mv::Data::OpListIterator op,
                    mv::Tiling& tiling, bool /*vertical_fusion_overlap*/ = false)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto childTiles = tiling.childTiles();

    // Attributes query based on blacklist
    // Batch stream, need only overwrite shape
    auto attrsToCopy = op->getAttrs({"shape"});
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
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv" || opType == "HwConvert")
        {
            auto inputTensor  = op->getInputTensor(0);
            auto outputTensor = op->getOutputTensor(0);

            auto inputQuantParams  = inputTensor->getQuantParams();
            auto outputQuantParams = outputTensor->getQuantParams();

            auto outputDType = outputTensor->getDType();

            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            auto slice = om.slice(op->getName() + "_slice" + tiling.getAxis() + std::to_string(split),
                                inputTensor,
                                sliceStart,
                                sliceShape);
            slice->setQuantParams(inputQuantParams);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if (opType == "MaxPool")
                newTensor = om.maxPool(streamingOpName,
                                slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                padding,
                                op->get<const bool>("exclude_pad"));

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                padding,
                                op->get<unsigned>("dilationFactor"));

            if (opType == "Conv")
                newTensor = om.conv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                padding,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"));

            if (opType == "HwConvert")
                newTensor = om.hwConvert(streamingOpName,
                                slice,
                                outputDType);

            newTensor->setDType(outputDType);
            newTensor->setQuantParams(outputQuantParams);

            slices.push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->getOutputTensor(0)->getDType();
            for (unsigned i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                auto inputQuantParams = inputTensor->getQuantParams();

                auto sliceShape = childTiles[split].getActivationShape();
                auto sliceStart = childTiles[split].getActivationStart();

                auto slice = om.slice(op->getName() + "_slice"  + tiling.getAxis() + std::to_string(split) + "_" + std::to_string(i),
                                inputTensor,
                                sliceStart,
                                sliceShape);
                slice->setQuantParams(inputQuantParams);
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices.push_back(slice);
            }

            auto quantParams = op->getOutputTensor(0)->getQuantParams();
            newTensor = om.eltwise(op->getName() + "_stream" + tiling.getAxis() + std::to_string(split),
                                   slices,
                                   eltwiseType);
            newTensor->setDType(originalDType);
            newTensor->setQuantParams(quantParams);
        }

        // Restore original out dtype, to account for mixed precision cases
        // where we don't want the same datatype for output as the input tensors
        newTensor->setDType(op->getOutputTensor(0)->getDType());
        auto newOp = om.getSourceOp(newTensor);
        newOp->set<bool>("shareWeights", true);

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

    auto outputTensor = op->getOutputTensor("output");
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

            out = newStreamFunc(om, om.getSourceOp(newTensors[split]), childTiles[split], false);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
            out = newTensors[split];
        final_outputs[split] = out;
    }

    auto quantParams = op->getOutputTensor(0)->getQuantParams();
    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    tiling.getAxis());
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(quantParams);
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

void findAndKeepTheParentWithHigherLevel(std::set<std::string> &previousOps, mv::OpModel& om)
{
    if (previousOps.size() < 2)
        return;
    std::set<std::string> toRemove;
    for (auto pOp = previousOps.begin(); pOp != previousOps.end(); ++pOp)
    {
        auto parentOp = om.getOp(*pOp);
        if (!parentOp->hasAttr("inputMaxLevel") ||
            !parentOp->get<bool>("inputMaxLevel"))
        {
            toRemove.insert(*pOp);
        }
    }

    for (auto &op : toRemove)
        previousOps.erase(op);

    return;
}

std::unordered_map<std::string, std::size_t> generateTailLevelMap(mv::OpModel& om)
{
    std::list<std::string> zero_in_degree_nodes[2UL];
    std::unordered_map<std::string, size_t> in_degree_map;
    std::unordered_map<std::string, size_t> task_level;
    size_t curr_depth = 0;
    // STEP-0: compute the in-degree's of all nodes //
    //NOTE: in_degree means the number of inputs of an op, and the pseudo data flows
    //if an op is zero_in_degree goes to zero_in_degree_nodes, like constants
    for (auto op_itr = om.opBegin(); op_itr != om.opEnd(); ++op_itr)
    {
        in_degree_map[ op_itr->getName() ] = op_itr->getInputTensor().size();
        if (op_itr->getInputTensor().size() == 0)
            zero_in_degree_nodes[0].push_back(op_itr->getName());
    }

    // NOTE: Topological sort according to zero_in_degree algorithm,
    // link: https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
    // STEP-1: populate the dpu-levels map, pretty much
    // takes the opmodel as a dag and provides the ops that are on which level
    // e.g. A->B->C , A->D then (2, {B,D} )
    while (!zero_in_degree_nodes[curr_depth%2UL].empty())
    {
        bool parity = ((curr_depth%2UL) == 1UL);
        for (auto zitr=zero_in_degree_nodes[parity].begin();
              zitr!=zero_in_degree_nodes[parity].end(); ++zitr)
        {
          // update the in-degree //
          mv::Data::OpListIterator zop_itr = om.getOp((*zitr));
          for (auto citr=zop_itr.leftmostChild(); citr!=om.opEnd(); ++citr)
          {
            std::string cop_name = citr->getName();
            auto ditr = in_degree_map.find(cop_name);
            if ( (ditr == in_degree_map.end()) || (ditr->second == 0UL) )
            {
                throw mv::RuntimeError("Streaming pass", "Missing entry in the in-degree map (or)"
                  " invalid in-degree for op= " + cop_name);
            }
            --(ditr->second);
            if (!(ditr->second))
            {
                zero_in_degree_nodes[!parity].push_back(cop_name);
                task_level[cop_name] = (curr_depth);
            }
          }
        }
        zero_in_degree_nodes[parity].clear();
        curr_depth++;
    }
    return task_level;
}

void computeStreamsForVerticalFusionNode(const std::string& opName, const std::vector<mv::Shape>& outputTileSizes,
    const std::vector<mv::Shape>& outputTileStarts, mv::ComputationModel& model, const std::vector<mv::Element>& strategies,
    std::unordered_map<std::string, std::vector<mv::Shape>>& previousOutputTileSizes,
    std::unordered_map<std::string, std::vector<mv::Shape>>& previousOutputTileStarts)
{
    mv::Element layerNameStrategy("streaming_strategy");
    for (auto strategy = strategies.begin(); strategy != strategies.end(); ++strategy)
    {
        auto strat = *strategy;
        std::string nodeName = (strat).get<std::string>("name_filter");
        if (nodeName == opName)
            layerNameStrategy = *strategy;
    }
    mv::OpModel om(model);
    auto opIt = om.getOp(opName);
    auto inputTensor = opIt->getInputTensor(0);
    auto inputShape = inputTensor->getShape();
    auto opType = opIt->getOpType();

    mv::Tiling masterTile;
    if((opType == "Conv") || (opType == "DepthwiseConv"))
    {
        auto kernelShape = opIt->getInputTensor(1)->getShape();
        masterTile = mv::Tiling(inputShape,kernelShape);
    }
    else
        masterTile = mv::Tiling(inputShape);

    auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");
    std::vector<mv::Tiling*> tiles = {&masterTile};
    auto verticalFusionApplyTiling = [opIt](mv::Element& split, mv::Tiling& tile, const std::vector<mv::Shape>& tileSizes,
        const std::vector<mv::Shape>& tilesStarts) -> std::vector<mv::Tiling>*
    {
        //the axis&split are stored in a map with key-> val .....
        //Don't want to if-then-else over all possible values of the axis...
        //the map should have only one key.. this is the draw-back of too generic mv::Element
        auto axis = split.attrsKeys()[0];
        auto numSplits = split.get<int>(axis);

        if(numSplits > 1)
        {
            tile.setAxis(axis);
            tile.resizeNumberOfTiles(numSplits);
            tile.generateTiling(opIt, true, tileSizes, tilesStarts);
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
            auto childTiles = verticalFusionApplyTiling(split,*tile, outputTileSizes, outputTileStarts);
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
        mv::Data::OpListIterator parentOp;
        if (opIt->getOpType() == "Eltwise")
        {
            std::set<std::string> previousOps;
            for (std::size_t input = 0; input < opIt->getInputTensor().size(); ++input)
            {
                if (!opIt->getInputTensor()[input]->isPopulated())
                    previousOps.insert(om.getSourceOp(opIt->getInputTensor()[input])->getName());
            }
            findAndKeepTheParentWithHigherLevel(previousOps, om);
            parentOp = om.getOp(*previousOps.begin());
        }
        else
            parentOp = om.getSourceOp(opIt->getInputTensor()[0]);

        auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile, false);

        if ((parentOp->hasAttr("verticalFusionSubgraphHead") && parentOp->get<bool>("verticalFusionSubgraphHead")) ||
                (parentOp->hasAttr("verticalFusion") && parentOp->get<bool>("verticalFusion")))
        {
            for (auto&& tile : masterTile.childTiles())
            {
                previousOutputTileStarts[parentOp->getName()].push_back(tile.getActivationStart());
                previousOutputTileSizes[parentOp->getName()].push_back(tile.getActivationShape());
            }
        }
        //NOTE: FlowSibling iterators seem to lose some sinks so they are replced...
        // reconnect children to subgraph
        std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;
        auto outputTensor = opIt->getOutputTensor()[0];
        for (auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
        {
            auto consumer = output.sink();
            std::size_t slot = 0;
            for (std::size_t input_idx = 0; input_idx < consumer->getInputTensor().size(); input_idx++)
                if (consumer->getInputTensor()[input_idx]->getName() == outputTensor->getName())
                    slot = input_idx;
            toReturn.push_back(std::make_pair(consumer, slot));
        }

        om.removeOp(opIt);
        for (unsigned j = 0; j < toReturn.size(); ++j)
        {
            toReturn[j].first->setInputTensor(result, toReturn[j].second, false);
            om.defineFlow(result, toReturn[j].first, toReturn[j].second);
        }
    }

    return;
}

void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element& passDesc,
                                mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    bool vertical_fusion = passDesc.hasAttr("vertical_fusion") ? passDesc.get<bool>("vertical_fusion"): false;
    bool yolo_v4 = om.getInput()->hasAttr("yolo_v4") ? om.getInput()->get<bool>("yolo_v4"): false;
    std::unordered_map<std::string, std::vector<mv::Shape>> previousOutputTileSizes;
    std::unordered_map<std::string, std::vector<mv::Shape>> previousOutputTileStarts;
    std::unordered_map<std::string, std::size_t> tailLevelMap;
    std::unordered_map<std::string, std::string> pivot_nodes;
    std::vector<std::string> secondTails;
    if (vertical_fusion)
    {
        tailLevelMap = generateTailLevelMap(om);
        //NOTE: the code below finds the op with the highest level for the ops that have multiple parents
        for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
        {
            std::string maxName, minName;
            if ((opIt->getOpType() == "Eltwise"
                 && opIt->hasAttr("verticalFusion")) || opIt->getOpType() == "Concat" || opIt->getOpType() == "ImplicitConcat")
            {
                maxName = om.getSourceOp(opIt->getInputTensor()[0])->getName();
                minName = om.getSourceOp(opIt->getInputTensor()[0])->getName();
                std::size_t maxLevel = tailLevelMap[om.getSourceOp(opIt->getInputTensor()[0])->getName()];
                std::size_t minLevel = tailLevelMap[om.getSourceOp(opIt->getInputTensor()[0])->getName()];
                mv::Data::OpListIterator parentOp;
                for (std::size_t inputIdx = 1; inputIdx < opIt->getInputTensor().size(); ++inputIdx)
                {
                    parentOp = om.getSourceOp(opIt->getInputTensor()[inputIdx]);
                    if (tailLevelMap[parentOp->getName()] > maxLevel)
                    {
                        maxLevel = tailLevelMap[parentOp->getName()];
                        maxName = parentOp->getName();
                    }
                    if (tailLevelMap[parentOp->getName()] < minLevel)
                    {
                        minLevel = tailLevelMap[parentOp->getName()];
                        minName = parentOp->getName();
                    }
                }
                om.getOp(maxName)->set<bool>("inputMaxLevel", true);
                om.getOp(minName)->set<bool>("inputLowLevel", true);
                if (opIt->hasAttr("overLappingSubgraphOpsIndex"))
                    om.getOp(minName)->set<std::size_t>("overLappingSubgraphOpsIndex", opIt->get<std::size_t>("overLappingSubgraphOpsIndex"));
            }
        }
    }

    //NOTE: NESTED STREAMING MEANS 2 LEVELS OF STREAMING, eg. HK, Stream Over H will stream
    //the input Tensor of the Op and then for every new Op have to stream it over K, which
    //means the weights will be repeated for the second level of streaming, this is why need
    //the data structures below...to create only one pair of nested slices
    for (auto layerNameStrategy : strategyList)
    {
        //NOTE: if we have vertical fusion we need to compute some extra overlaps
        //so we will start only with the tails and the streaming ops
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        //NOTE: ensure that the op with that name exists in the opModel
        if (!om.checkOp(nodeName))
        {
            pass.log(mv::Logger::MessageType::Debug, nodeName + " is not present in model, skipping streaming");
            continue;
        }

        //NOTE: ensure that the op with that name exists in the opModel
        auto opIt =  om.getOp(nodeName);
        std::set<std::string> previousOps = {};
        bool nextOpConcat = false;

        if (vertical_fusion)
        {
            if ((opIt->hasAttr("verticalFusionSubgraphHead") && opIt->get<bool>("verticalFusionSubgraphHead")) ||
                (opIt->hasAttr("verticalFusion") && opIt->get<bool>("verticalFusion")))
                continue;

            if (opIt->getOpType() != "Output")
            {
                std::string nextOpType = (mv::findSinkLayers(dm, opIt->getOutputTensor()[0])[0])->getOpType();
                nextOpConcat = (nextOpType == "Concat" || nextOpType == "ImplicitConcat");
            }
        }
        if (vertical_fusion && (opIt->hasAttr("verticalFusionSubgraphTail") && opIt->get<bool>("verticalFusionSubgraphTail")))
        {
            //NOTE: previous Op will be needed only for the tail vertical fusion ops
            auto inputs = opIt->getInputTensor().size();
            for (std::size_t input = 0; input < inputs; ++input)
            {
                auto inputTensor = opIt->getInputTensor()[input];
                if (!inputTensor->isPopulated())
                {
                    auto previousOp = om.getSourceOp(inputTensor);
                    //NOTE: need to compute the tiles from all the intermiedate first
                    bool previousIsHead = (previousOp->hasAttr("verticalFusionSubgraphHead") && previousOp->get<bool>("verticalFusionSubgraphHead"));
                    bool previousHasUniqueChild = (mv::findSinkLayers(dm, previousOp->getOutputTensor()[0]).size() == 1);
                    if ((previousIsHead && previousHasUniqueChild) || !previousIsHead)
                        previousOps.insert(previousOp->getName());
                }
                //NOTE: if tail is an eltwise, previousOps need to contain only the one op that
                // will have the smaller tiles which means the one with bigger level in the graph
                std::set<std::string> tempPreviousOps = previousOps;
                findAndKeepTheParentWithHigherLevel(previousOps, om);
            }
            //NOTE: when the tail is followed by concat we need to compute the tiles only for the op with the highest level
            if (nextOpConcat)
            {
                if (!opIt->hasAttr("inputMaxLevel") || !opIt->get<bool>("inputMaxLevel"))
                {
                    opIt->set<bool>("noRecusionVF", true);
                    secondTails.push_back(opIt->getName());
                    pivot_nodes[opIt->getName()] = (om.getSourceOp(opIt->getInputTensor()[0]))->getName();
                }
            }
        }
        std::string opType = opIt->getOpType();

        //For now do streaming pass only for the DPU layers
        if ((opType != "Conv") && (opType != "DepthwiseConv") && (opType != "MaxPool") && !opIt->isEltwiseTypeOp())
            continue;

        std::size_t alignment = 1;
        if(passDesc.hasAttr("alignment"))
            alignment = passDesc.get<int>("alignment");

        auto inputTensor = opIt->getInputTensor(0);
        auto inputShape = inputTensor->getShape();

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

        if ((!opIt->hasAttr("noRecusionVF") || !opIt->get<bool>("noRecusionVF")))
        {
            auto applyTiling = [opIt, alignment, pass](mv::Element& split, mv::Tiling& tile) -> std::vector<mv::Tiling>*
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
                    tile.setAlignment(alignment);
                    tile.resizeNumberOfTiles(numSplits);
                    tile.generateTiling(opIt, false);
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
                auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile, false);
                if (vertical_fusion && (!nextOpConcat || (!opIt->hasAttr("noRecusionVF") || !opIt->get<bool>("noRecusionVF"))))
                {
                    for (auto previous = previousOps.begin(); previous != previousOps.end(); ++previous)
                    {
                        auto pOp = om.getOp(*previous);
                        if (!pOp->hasAttr("outputTilesComputed") ||
                            (pOp->hasAttr("outputTilesComputed") && !pOp->get<bool>("outputTilesComputed")))
                        {
                            for (auto&& tile : masterTile.childTiles())
                            {
                                previousOutputTileStarts[pOp->getName()].push_back(tile.getActivationStart());
                                previousOutputTileSizes[pOp->getName()].push_back(tile.getActivationShape());
                            }
                        }
                        pOp->set<bool>("outputTilesComputed", true);
                    }
                }
                //NOTE: FlowSibling iterators seem to lose some sinks so they are replced...
                // reconnect children to subgraph
                std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;
                auto outputTensor = opIt->getOutputTensor()[0];
                for (auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
                {
                    auto consumer = output.sink();
                    std::size_t slot = 0;
                    for (std::size_t input_idx = 0; input_idx < consumer->getInputTensor().size(); input_idx++)
                        if (consumer->getInputTensor()[input_idx]->getName() == outputTensor->getName())
                            slot = input_idx;
                    toReturn.push_back(std::make_pair(consumer, slot));
                }

                om.removeOp(opIt);
                for (std::size_t j = 0; j < toReturn.size(); ++j)
                {
                    toReturn[j].first->setInputTensor(result, toReturn[j].second, false);
                    om.defineFlow(result, toReturn[j].first, toReturn[j].second);
                }
            }
        }
    }

    if (vertical_fusion)
    {
        auto sortedOps = om.topologicalSort();
        std::vector<std::string> verticalFusionHeadOps;
        for (auto opIt = sortedOps.begin(); opIt != sortedOps.end(); ++opIt)
        {
            auto op = *opIt;
            if ((op->hasAttr("verticalFusionSubgraphHead") && op->get<bool>("verticalFusionSubgraphHead")) ||
                (op->hasAttr("verticalFusion") && op->get<bool>("verticalFusion")))
                verticalFusionHeadOps.push_back(op->getName());
        }
        if (!verticalFusionHeadOps.empty())
        {
            std::reverse(verticalFusionHeadOps.begin(), verticalFusionHeadOps.end());

            auto root = verticalFusionHeadOps.begin();
            auto last = verticalFusionHeadOps.back();

            if (*root == last)
                computeStreamsForVerticalFusionNode(*root, previousOutputTileSizes[*root], previousOutputTileStarts[*root], model, strategyList,
                    previousOutputTileSizes, previousOutputTileStarts);
            else
            {
                auto next = root + 1;
                while (!om.getOp(last)->hasAttr("outputTilesComputed"))
                {
                    computeStreamsForVerticalFusionNode(*root, previousOutputTileSizes[*root], previousOutputTileStarts[*root], model, strategyList,
                        previousOutputTileSizes, previousOutputTileStarts);
                    om.getOp(*next)->set<bool>("outputTilesComputed", true);
                    root = next;
                    next = next + 1;
                }
                om.getOp(*root)->set<bool>("outputTilesComputed", true);
                computeStreamsForVerticalFusionNode(*root, previousOutputTileSizes[last], previousOutputTileStarts[last], model, strategyList,
                    previousOutputTileSizes, previousOutputTileStarts);
            }
        }
        //NOTE: in the end compute the tiling for the second last, tail, now we will copy them cause in yolov4 second tail is neutral
        //normally a function computing output tiles from input ones would be needed
        if (yolo_v4)
        {
            for (auto opName = secondTails.begin(); opName != secondTails.end(); ++opName)
            {
                auto opIt = om.getOp(*opName);
                //NOTE: if we have vertical fusion we need to compute some extra overlaps
                //so we will start only with the tails and the streaming ops
                if (opIt->hasAttr("noRecusionVF") && opIt->get<bool>("noRecusionVF"))
                {
                    auto inputShape = opIt->getInputTensor()[0]->getShape();
                    auto kernelShape = opIt->getInputTensor()[1]->getShape();
                    std::vector <mv::Shape> inputTileStarts = previousOutputTileStarts[pivot_nodes[opIt->getName()]];
                    std::vector <mv::Shape> inputTileSizes = previousOutputTileSizes[pivot_nodes[opIt->getName()]];
                    mv::Tiling masterTile;
                    masterTile = mv::Tiling(inputShape, kernelShape);

                    std::vector<mv::Tiling*> tiles = {&masterTile};
                    auto applySecondTailTiling = [opIt, pass](mv::Tiling& tile, const std::vector<mv::Shape>& tileSizes,
                        const std::vector<mv::Shape>& tilesStarts) -> std::vector<mv::Tiling>*
                    {
                        //the axis&split are stored in a map with key-> val .....
                        //Don't want to if-then-else over all possible values of the axis...
                        //the map should have only one key.. this is the draw-back of too generic mv::Element
                        auto numSplits = tilesStarts.size();

                        if(numSplits > 1)
                        {
                            tile.setAxis("H");
                            tile.setAlignment(false);
                            tile.resizeNumberOfTiles(numSplits);
                            tile.generateTiling(opIt, false, tileSizes, tilesStarts, true);
                            return &tile.childTiles();
                        }
                        else
                        {
                            return nullptr;
                        }
                    };

                    std::vector<mv::Tiling*> newChildTiles(0);
                    for(auto tile : tiles)
                    {
                        auto childTiles = applySecondTailTiling(*tile, inputTileSizes, inputTileStarts);
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
                    if(masterTile.childTiles().size() > 1)
                    {
                        auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile, true);
                        std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;
                        auto outputTensor = opIt->getOutputTensor()[0];
                        for (auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
                        {
                            auto consumer = output.sink();
                            std::size_t slot = 0;
                            for (std::size_t input_idx = 0; input_idx < consumer->getInputTensor().size(); input_idx++)
                                if (consumer->getInputTensor()[input_idx]->getName() == outputTensor->getName())
                                    slot = input_idx;
                            toReturn.push_back(std::make_pair(consumer, slot));
                        }

                        om.removeOp(opIt);
                        for (std::size_t j = 0; j < toReturn.size(); ++j)
                        {
                            toReturn[j].first->setInputTensor(result, toReturn[j].second, false);
                            om.defineFlow(result, toReturn[j].first, toReturn[j].second);
                        }
                    }
                }
            }
        }
    }
    
    /// Merge continuous slice ops
    auto sliceOps = om.getOps("Slice");
    for(auto& sliceOp: sliceOps)
    {
        if(sliceOp->hasAttr("dilatedSlice") && sliceOp->get<bool>("dilatedSlice"))
            continue;
        auto children = mv::findSinkLayers(dm, sliceOp->getOutputTensor(0));
        bool allSlice = true;
        for(auto& op: children)
        {
            if(allSlice){
                allSlice = allSlice && (op->getOpType() == "Slice");
            }else {
                break;
            }
        }

        if(allSlice)
        {
            auto inputTensor = sliceOp->getInputTensor(0);
            auto begin = sliceOp->get<mv::Shape>("begin");
            for(auto& op: children)
            {
                auto childBegin = op->get<mv::Shape>("begin");
                auto newBegin = childBegin + begin;
                op->set<mv::Shape>("begin", newBegin);
                om.defineFlow(inputTensor, op, 0);
                op->setInputTensor(inputTensor, 0, false);
            }

            om.removeOp(sliceOp);            
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
    auto opIterator = om.opBegin();
    while (opIterator != om.opEnd())
    {
        std::string opType = opIterator->getOpType();

        if (opType == "Slice" && opIterator->getInputTensor(0)->isPopulated())
        {
            auto inTensorSlice = opIterator->getInputTensor(0);
            removeConstantsSet.insert(om.getSourceOp(inTensorSlice)->getName());
            auto outTensorSlice = opIterator->getOutputTensor(0);
            auto parentOpIt = om.getSourceOp(opIterator->getInputTensor(0));
            auto shape = outTensorSlice->getShape();
            auto quantParams = outTensorSlice->getQuantParams();
        
            auto newConstant = om.constantDataElement(opIterator->getName() + "_weights",
                                                      outTensorSlice->getData(), shape,
                                                      outTensorSlice->getDType(), outTensorSlice->getOrder());
            newConstant->setQuantParams(quantParams);
            newConstant->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
            auto constantOp = om.getSourceOp(newConstant);
            if(opIterator->hasAttr("opId"))
            {
                unsigned currentOpId = opIterator->get<unsigned>("opId");
                constantOp->set<unsigned>("opId", currentOpId);
            }
            auto copyIterator = opIterator;
            ++opIterator;
            copyIterator = operationsReplacement(parentOpIt, newConstant, om, copyIterator);
        }
        else
            ++opIterator;
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
