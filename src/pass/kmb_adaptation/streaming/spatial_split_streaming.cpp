#include "math.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/tensor/tiling.hpp"

static void streamingOperations(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry&,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StreamingTiling)
        .setFunc(streamingTilingFcn)
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

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef>>& thisGraphStrategy, std::unordered_map<std::string, bool> &createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp);
mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef>>& thisGraphStrategy, std::unordered_map<std::string, bool> &createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp);

std::map<std::string, std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, mv::Tiling&, std::map<std::string, std::vector<opStreamingSplitDef>>&, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)>> streamSplit =
{
//    {"W",solveSpatialTiling},
    {"H",solveSpatialTiling},
    {"K",solveWeightsTiling} //NOTE::Only Convolution is supported for SoK now
};


std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, mv::Tiling&, std::map<std::string, std::vector<opStreamingSplitDef>> &, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)> convSpatialTiling = solveSpatialTiling;
std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, mv::Tiling&, std::map<std::string, std::vector<opStreamingSplitDef>> &, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)> convOutChannelTiling = solveWeightsTiling;

static void setStreamingStrategy(const mv::pass::PassEntry &pass, mv::ComputationModel &model, std::map<std::string, std::vector<opStreamingSplitDef>> &thisGraphStrategy)
{
    // get ops to split and number of splits from descriptor
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        std::cout << "No strategy defined in JSON" << std::endl;
        pass.log(mv::Logger::MessageType::Info, "No custom streaming strategy provided");
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

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op,
    mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef> > &thisGraphStrategy,
    std::unordered_map<std::string, bool>& createSlicesPerStream,
    std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    auto inputTensor = op->getInputTensor("data");
    auto kernelTensor = op->getInputTensor("weights");
    auto outputTensor = op->getOutputTensor("output");
    auto inputTensor2Conv = inputTensor ;
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
        if ((!foundOpName) || (foundOpName && foundIt->second))
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
        auto conv = om.conv(inputTensor2Conv,
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
        newOp->set<unsigned>("opId",opId);
        newOp->set<std::string>("splitStrategy", splitStrategy);
        newOp->set<bool>("inputActivationSparsity", op->get<bool>("inputActivationSparsity"));
        newOp->set<bool>("outputActivationSparsity", op->get<bool>("outputActivationSparsity"));
        newOp->set<bool>("weightsSparsity", op->get<bool>("weightsSparsity"));
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
        if(childTiles[split].childTiles().size() > 1)
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
            out = (streamSplit[childTiles[split].getAxis()])(om,om.getSourceOp(convs[split]),childTiles[split], thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);
            om.removeOp( om.getSourceOp(convs[split]));
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

    return concat;
}

mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string,
                std::vector<opStreamingSplitDef> > &thisGraphStrategy, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto outputTensor = op->getOutputTensor("output");
    auto opId = op->get<unsigned>("opId");
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

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
            auto inputTensor = op->getInputTensor(0);
            //NOTE: NESTED STREAM NEEDS SLICE OPS ONLY FOR THE FIRST PART AND RE-USE
            mv::Data::TensorIterator slice;
            auto foundIt = createSlicesPerStream.find(op->getName());
            bool foundOpName = false;
            //NOTE:THIS OP NAME DOES NOT EXIST SO NO NESTED...GO AND BUILD SLICE
            if (foundIt != createSlicesPerStream.end())
                foundOpName = true;
            if ((!foundOpName) || (foundOpName && foundIt->second))
            {
                slice = om.slice(inputTensor,
                                    childTiles[split].getStartCoord(),
                                    childTiles[split].getSize(),
                                    inputTensor->get<mv::QuantizationParams>("quantParams"),
                                    op->getName() + "_sliceH_" + std::to_string(split));
                storeExistingSlice(inputTensor->getName(), split, slice, name_firstStream_sliceOp);
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
            }
            else
            {
                std::pair<std::string, unsigned> keyPair;
                keyPair.first = inputTensor->getName();
                keyPair.second = split;
                slice = name_firstStream_sliceOp[keyPair];
            }
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
        else if (opType == "Add" || opType == "Subtract" || opType == "Multiply")
        {
            for (auto i = 0; i < 2; i++)
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
            auto addFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const mv::DType& dType, const std::string& s){ return om.add(vec, dType, quantParams, s);};
            auto subFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const mv::DType& dType, const std::string& s){ return om.subtract(vec, dType, quantParams, s);};
            auto multFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const mv::DType& dType, const std::string& s){ return om.multiply(vec, dType, quantParams, s);};

            auto dpuTaskMap = std::map<std::string, std::function<mv::Data::TensorIterator (std::vector< mv::Data::TensorIterator >&, const mv::QuantizationParams&, const mv::DType&, const std::string&)>>
                                                    {{"Add", addFcn},
                                                    {"Subtract", subFcn},
                                                    {"Multiply", multFcn}};
            auto dpuElementWiseFunctor = (dpuTaskMap.at(opType));
            newTensor = dpuElementWiseFunctor(slices[split], op->get<mv::QuantizationParams>("quantParams"),
                                              op->get<mv::DType>("dType"), streamingOpName);
        }

        //NOTE: Nested streaming case
        if (thisGraphStrategy[op->getName()].size() > 1)
        {
            thisGraphStrategy[streamingOpName].insert(thisGraphStrategy[streamingOpName].begin(), thisGraphStrategy[op->getName()].begin() + 1,
                    thisGraphStrategy[op->getName()].end());
            //NOTE:: If we go HK streaming...
            if (split == 0)
                createSlicesPerStream[streamingOpName] = true;
            else
                createSlicesPerStream[streamingOpName] = false;
        }
        auto newOp = om.getSourceOp(newTensor);

        if (op->hasAttr("bias"))
        {
            auto biasTensorName = op->get<std::string>("bias");
            om.addAttr(newOp, "bias", biasTensorName);
        }

        newOp->set<unsigned>("opId", opId);
        newOp->set<std::string>("splitStrategy", splitStrategy);
        newOp->set<bool>("inputActivationSparsity", op->get<bool>("inputActivationSparsity"));
        newOp->set<bool>("outputActivationSparsity", op->get<bool>("outputActivationSparsity"));
        newOp->set<bool>("weightsSparsity", op->get<bool>("weightsSparsity"));
        if (op->hasAttr("postOpType"))
        {
            newOp->set<std::string>("postOpType", op->get<std::string>("postOpType"));
            if (newOp->get<std::string>("postOpType") == "LeakyRelu")
                newOp->set<double>("alpha", op->get<double>("alpha"));
        }
        else if (op->hasAttr("postOpTypes"))
        {
            newOp->set<std::vector<std::string>>("postOpTypes", op->get<std::vector<std::string>>("postOpTypes"));
            std::vector<std::string> postOpTypes = op->get<std::vector<std::string>>("postOpTypes");

            if (op->getOutputTensor(0)->getDType() ==  mv::DType("Float16"))
            {
                newOp->set<double>("minimum", op->get<double>("minimum"));
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Maximum") != postOpTypes.end())
                    newOp->set<double>("maximum", op->get<double>("maximum"));
            }
            else
            {
                newOp->set<int64_t>("minimum", op->get<int64_t>("minimum"));
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Maximum") != postOpTypes.end())
                    newOp->set<int64_t>("maximum", op->get<int64_t>("maximum"));
            }
        }

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
        if (childTiles[split].childTiles().size() > 1)
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
            out = (streamSplit[childTiles[split].getAxis()])(om, om.getSourceOp(convs[split]), childTiles[split], thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);
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
    return concat;
}

void streamingOperations(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element&,
                                mv::Element&)
{

    mv::OpModel om(model);
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
        bool isElementWise = (opType == "Add" || opType == "Subtract" || opType == "Multiply");
        if (opType == "Conv" || opType == "DepthwiseConv" ||  (opType == "MaxPool") || isElementWise)
        {
            int numberOfSplits = thisOpStrategy[0].numSplits ;
            std::string axisToSplit = thisOpStrategy[0].axis ;
            mv::Tiling masterTile(axisToSplit, numberOfSplits);
            mv::Shape masterSize;

            if (axisToSplit == "K")
            {
                masterTile.setSize(opIt->getInputTensor("weights")->getShape());
                masterTile.generateWeightsTiling();
            }
            else
            {
                masterTile.setSize(opIt->getInputTensor("data")->getShape());
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
