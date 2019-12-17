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
    }
}

mv::Data::OpListIterator operationsReplacement(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
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

std::tuple<mv::Data::TensorIterator, mv::Data::TensorIterator,mv::Data::TensorIterator> solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef>>& thisGraphStrategy, std::unordered_map<std::string, bool> &createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp);
std::tuple<mv::Data::TensorIterator, mv::Data::TensorIterator,mv::Data::TensorIterator> solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef>>& thisGraphStrategy, std::unordered_map<std::string, bool> &createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp);

std::map<std::string, std::function<std::tuple<mv::Data::TensorIterator, mv::Data::TensorIterator,mv::Data::TensorIterator>(mv::OpModel&, mv::Data::OpListIterator, mv::Tiling&, std::map<std::string, std::vector<opStreamingSplitDef>>&, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)>> streamSplit =
{
//    {"W",solveSpatialTiling},
    {"H",solveSpatialTiling},
    {"K",solveWeightsTiling} //NOTE::Only Convolution is supported for SoK now
};

static void setStreamingStrategy(const mv::pass::PassEntry &pass, mv::ComputationModel &model, std::map<std::string, std::vector<opStreamingSplitDef>> &thisGraphStrategy)
{
    // get ops to split and number of splits from descriptor
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        std::cout << "No strategy defined in JSON" << std::endl;
        pass.log(mv::Logger::MessageType::Debug, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    // each s refers to the name of an op, from the JSON strategy list
    for (auto layerNameStrategy : strategyList)
    {
        std::vector<opStreamingSplitDef> opxSplits;
        bool nodeHasSplit = false;
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");
        for (std::size_t i = 0; i < splitList.size(); i++)
        {
            opStreamingSplitDef opxSplitx;
            if (splitList[i].hasAttr("H"))
            {
                if (splitList[i].get<int>("H") > 1)
                {
                    opxSplitx.axis = "H";
                    opxSplitx.numSplits = splitList[i].get<int>("H");
                    opxSplits.push_back(opxSplitx);
                    nodeHasSplit = true;
                    std::cout << "Streaming for node: " << nodeName << " has stream H = " << opxSplitx.numSplits << std::endl ;
                }
            }
            //NOTE:: Streaming over width, channels are not used
//            else if (splitList[i].hasAttr("W"))
//            {
//                if (splitList[i].get<int>("W")>1)
//                {
//                    opxSplitx.axis = "W";
//                    opxSplitx.numSplits = splitList[i].get<int>("W");
//                    opxSplits.push_back(opxSplitx);
//                    nodeHasSplit=true;
//                }
//            }
//            else if (splitList[i].hasAttr("C"))
//            {
//                if (splitList[i].get<int>("C")>1)
//                {
//                    opxSplitx.axis = "C";
//                    opxSplitx.numSplits = splitList[i].get<int>("C");
//                    opxSplits.push_back(opxSplitx);
//                    nodeHasSplit=true;
//                }
//            }
            if (splitList[i].hasAttr("K"))
            {
                if (splitList[i].get<int>("K") > 1)
                {
                    opxSplitx.axis = "K";
                    opxSplitx.numSplits = splitList[i].get<int>("K");
                    opxSplits.push_back(opxSplitx);
                    if (nodeHasSplit)
                    {
                        std::reverse(opxSplits.begin(),opxSplits.end());
                    }

                    nodeHasSplit = true;
                    std::cout << "Streaming for node: " << nodeName << " has stream K = " << splitList[i].get<int>("K") << std::endl ;
                }
            }
        }
        if (nodeHasSplit)
            thisGraphStrategy.insert(std::pair<std::string, std::vector<opStreamingSplitDef>>(nodeName, opxSplits));
    }
}

void storeExistingSlice(std::string opName, unsigned streamId, mv::Data::TensorIterator slice,
    std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator>& name_firstStream_sliceOp)
{
    std::pair<std::string, unsigned> keyPair;
    keyPair.first = opName;
    keyPair.second = streamId;
    name_firstStream_sliceOp[keyPair] = slice;
}

std::tuple<mv::Data::TensorIterator, mv::Data::TensorIterator,mv::Data::TensorIterator> solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op,
    mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef> > &thisGraphStrategy,
    std::unordered_map<std::string, bool>& createSlicesPerStream,
    std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto inputTensor2Conv = inputTensor ;
    auto kernelTensor = op->getInputTensor(1);
    auto outputTensor = op->getOutputTensor(0);
    bool nestedLayerStreaming = false;


    auto attrsToCopy = op->getAttrs({"stride", "padding", "shape", "bias"});

    mv::QuantizationParams quantParams = {{},{},{},{}};
    if(inputTensor->hasAttr("quantParams"))
        quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");
    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();
    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    size_t biasStartIndex = 0;
    size_t biasEndIndex = 0;
    std::string splitStrategy = op->get<std::string>("splitStrategy");

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator slice;
        bool foundOpName = false;
        //NOTE:THIS OP NAME DOES NOT EXIST SO NO NESTED...GO AND BUILD SLICE
        auto foundIt = createSlicesPerStream.find(op->getName());
        if (foundIt != createSlicesPerStream.end())
            foundOpName = true;
        if ((!foundOpName) //no nesting case
            || (foundOpName && foundIt->second))
        {
            if (kernelTensor->hasAttr("quantParams"))
            {
                slice = om.slice(kernelTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                kernelTensor->get<mv::QuantizationParams>("quantParams"),
                                kernelTensor->getName() + inputTensor->getName() + "_sliceK_" + std::to_string(split));
            }
            else
            {
                slice = om.slice(kernelTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                {{}, {}, {}, {}},
                                kernelTensor->getName() + "_sliceK_" + std::to_string(split));
            }
            storeExistingSlice(kernelTensor->getName(), split, slice, name_firstStream_sliceOp);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);
        }
        else
        {
            std::pair<std::string, unsigned> keyPair;
            keyPair.first = kernelTensor->getName();
            keyPair.second = split;
            slice = name_firstStream_sliceOp[keyPair];
        }
        std::string streamingOpName = op->getName() + "_split_" + std::to_string(split);
        auto conv = om.conv(inputTensor,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);
        //NOTE: Nested streaming case KH
        if (thisGraphStrategy[op->getName()].size() > 1)
        {
            nestedLayerStreaming = true;
            thisGraphStrategy[streamingOpName].insert(thisGraphStrategy[streamingOpName].begin(), thisGraphStrategy[op->getName()].begin() + 1,
                    thisGraphStrategy[op->getName()].end());
            if (split == 0)
                createSlicesPerStream[streamingOpName] = true;
            else
                createSlicesPerStream[streamingOpName] = false;
        }

        if (op->hasAttr("bias"))
        {
            auto tileSize = childTiles[split].getSize()[axisToSplit];
            biasStartIndex = biasEndIndex;
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
        convs[split] = conv;
        bool enableSerialStreaming = true;
        if ((split>0)&&(enableSerialStreaming))
            cm.defineFlow(om.getSourceOp(convs[split-1]), om.getSourceOp(convs[split]));
    }

    kernelTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    for(unsigned split = 0 ; split < number_of_splits; split++)
    {
        mv::Tensor::MemoryLocation inputLocation;
        mv::Tensor::MemoryLocation outputLocation;
        if(nestedLayerStreaming)
        {
            inputLocation.relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            inputLocation.force();
            outputLocation.force();
        }
        slices[split]->set<mv::Tensor::MemoryLocation>("Location", inputLocation);
        convs[split]->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
    }

    for(unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if(childTiles[split].childTiles().size() > 1)
        {
            auto temp = (streamSplit[childTiles[split].getAxis()])(om,om.getSourceOp(convs[split]),childTiles[split], thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);
            out = std::get<2>(temp);
            om.removeOp(om.getSourceOp(convs[split]));
        }
        else
            out = convs[split];
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

    return std::make_tuple(convs[0], convs[number_of_splits-1], concat);
}

std::tuple<mv::Data::TensorIterator, mv::Data::TensorIterator,mv::Data::TensorIterator>  solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string,
                std::vector<opStreamingSplitDef> > &thisGraphStrategy, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    bool nestedLayerStreaming = false;
    auto outputTensor = op->getOutputTensor("output");
    auto opId = op->get<unsigned>("opId");
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    // NOTE: In the streaming case, we can't just blindly copy everything like we
    // do in the DPUTask conversion case. We have to overwrite shape, padding, etc.
    auto attrsToCopy = op->getAttrs({"stride", "padding", "shape"});

    std::vector<mv::Shape> spatial_indexes(number_of_splits);
    std::vector<std::vector<mv::Data::TensorIterator>> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
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

//    if (axisToSplit == mv::Shape::getAxis("W"))
//    {
//        startPad[1] = 0;
//        endPad[0] = 0;
//        middlePad[0] = 0;
//        middlePad[1] = 0;
//    }
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
        std::string streamingOpName = op->getName() + "_split_" + std::to_string(split);
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv")
        {
            //for nested streaming we will duplicate the H slices, since the reason we streamed is that things
            // dont fit in CMX. If we dont duplicate the H slices, then all the H slices will have to stay in CMX
            // till we are done with all the first layer nesting (they are used by all the K branches).
            // it's either this or we add a mechanism to deallocate and reallocate them, but this solution
            // seems to be cheaper.
            auto inputTensor = op->getInputTensor(0);
            mv::Data::TensorIterator slice = om.slice(inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_sliceH_" + std::to_string(split));
            storeExistingSlice(inputTensor->getName(), split, slice, name_firstStream_sliceOp);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if (opType == "MaxPool")
                newTensor = om.maxPool(slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                currentPad,
                                op->get<const bool>("exclude_pad"),
                                op->get<std::string>("auto_pad"),
                                op->get<std::string>("rounding_type"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);

            if (opType == "Conv")
                newTensor = om.conv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                streamingOpName);
            slices[split].push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->get<mv::DType>("dType");
            for (auto i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);

                auto slice = om.slice(inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_sliceH_" + std::to_string(split) + "_" + std::to_string(i));
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices[split].push_back(slice);
            }

            newTensor = om.eltwise(slices[split], eltwiseType, originalDType, op->get<mv::QuantizationParams>("quantParams"), op->getName() + "_split_" + std::to_string(split));
        }

        //NOTE: Nested streaming case
        if (thisGraphStrategy[op->getName()].size() > 1)
        {
            nestedLayerStreaming = true;
            thisGraphStrategy[streamingOpName].insert(thisGraphStrategy[streamingOpName].begin(), thisGraphStrategy[op->getName()].begin() + 1,
                    thisGraphStrategy[op->getName()].end());
            //NOTE:: If we go HK streaming...
            if (split == 0)
                createSlicesPerStream[streamingOpName] = true;
            else
                createSlicesPerStream[streamingOpName] = false;
        }
        auto newOp = om.getSourceOp(newTensor);

        newOp->setAttrs(attrsToCopy);
        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated

        convs[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(convs[split-1]), om.getSourceOp(convs[split]));
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;
    for (unsigned split = 0 ; split < number_of_splits; split++)
    {
        auto numInputs = slices[split].size();
        std::vector<mv::Tensor::MemoryLocation> inputLocation(numInputs);
        mv::Tensor::MemoryLocation outputLocation;
        if (nestedLayerStreaming)
        {
            for (std::size_t i = 0; i < numInputs; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                inputLocation[i].relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            }
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            for (std::size_t i = 0; i < numInputs; i++)
            {
                inputLocation[i].relocate(mv::Tensor::MemoryLocation::NNCMX);
                inputLocation[i].force();
            }
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.force();
        }
        for (std::size_t i = 0; i < numInputs; i++)
            slices[split][i]->set<mv::Tensor::MemoryLocation>("Location", inputLocation[i]);
        convs[split]->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            auto temp = (streamSplit[childTiles[split].getAxis()])(om, om.getSourceOp(convs[split]), childTiles[split], thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);
            out = std::get<2>(temp);
            om.removeOp(om.getSourceOp(convs[split]));
        }
        else
            out = convs[split];
        final_outputs[split] = out;
    }
    std::vector<mv::Shape> final_outputs_deb(number_of_splits);
    for (std::size_t i=0; i < number_of_splits; ++i)
        final_outputs_deb[i] = final_outputs[i]->getShape();

    auto concat = om.concat(final_outputs,
                    tiling.getAxis(),
                    op->get<mv::DType>("dType"),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
    return std::make_tuple(convs[0], convs[number_of_splits-1], concat);
}

void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element&,
                                mv::Element&)
{

    mv::OpModel om(model);
    mv::ControlModel cm(model);
    std::map<std::string, std::vector<opStreamingSplitDef>> thisGraphStrategy;
    setStreamingStrategy(pass, model, thisGraphStrategy);
    std::vector<opStreamingSplitDef> thisOpStrategy;

    //NOTE: NESTED STREAMING MEANS 2 LEVELS OF STREAMING, eg. HK, Stream Over H will stream
    //the input Tensor of the Op and then for every new Op have to stream it over K, which
    //means the weights will be repeated for the second level of streaming, this is why need
    //the data structures below...to create only one pair of nested slices
    std::unordered_map<std::string, bool> createSlicesPerStream = {};
    std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> name_firstStream_sliceOp;

    for (auto layerNameStrategy: thisGraphStrategy)
    {
        std::string nodeName = layerNameStrategy.first;
        //NOTE: Graph optimizer will never do that but needs to be her for manual Scheduling
        if (!om.checkOp(nodeName))
        {
            pass.log(mv::Logger::MessageType::Error, nodeName + " is not present in model, skipping streaming");
            continue;
        }
        auto opIt =  om.getOp(nodeName);
        thisOpStrategy = thisGraphStrategy[nodeName];
        std::string opType = opIt->getOpType();

        if ((opType == "Conv" || opType == "DepthwiseConv" ||  (opType == "MaxPool") || (opType == "Eltwise")))
        {
            int numberOfSplits = thisOpStrategy[0].numSplits ;
            std::string axisToSplit = thisOpStrategy[0].axis ;
            mv::Tiling masterTile(axisToSplit, numberOfSplits);
            mv::Shape masterSize;

            if (axisToSplit == "K")
            {
                masterTile.setSize(opIt->getInputTensor(1)->getShape());
                masterTile.generateWeightsTiling();
            }
            else
            {
                masterTile.setSize(opIt->getInputTensor(0)->getShape());
                masterTile.generateSpatialTiling(opIt);
            }

            auto sourceTensor = opIt->getInputTensor(0);
            auto parentOpIt = om.getSourceOp(sourceTensor);
            auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile,
                               thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);

            // reconnect children to subgraph
            std::vector<mv::Data::OpListIterator> opsToLink;
            std::vector<std::size_t> inputSlots;
            for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                opsToLink.push_back(sinkFlow.sink());
                inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            }

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));

            om.removeOp(opIt);
            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                opsToLink[j]->setInputTensor(std::get<2>(result), inputSlots[j], false);
                om.defineFlow(std::get<2>(result), opsToLink[j], inputSlots[j]);
            }

            setInputControlFlow(cm, cm.switchContext(om.getSourceOp(std::get<0>(result))), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(om.getSourceOp(std::get<1>(result))), outputControlFlows);
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
