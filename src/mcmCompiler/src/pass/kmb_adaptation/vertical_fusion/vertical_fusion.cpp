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
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void recognizeVerticalFusionPatternsFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void verticalFusionTransformationFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void validateVerticalAdds(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(RecognizeVerticalFusionPatterns)
        .setFunc(recognizeVerticalFusionPatternsFcn)
        .setDescription(
                "Recognizes the vertical fusion subgraphs, at least 2 following ops that are streaming on H and are spilling to concatenate.");
    }

    namespace pass
    {
        MV_REGISTER_PASS(VerticalFusionTransformation)
        .setFunc(verticalFusionTransformationFcn)
        .setDescription(
                "Transforms the subgraphs after streaming to vertical fusion ones.");
    }

    namespace pass
    {
        MV_REGISTER_PASS(ValidateVerticalAdds)
        .setFunc(validateVerticalAdds)
        .setDescription(
                "Transforms the subgraphs after streaming to vertical fusion ones.");
    }
}

///////////////////////// PASS STATIC PARAMETERS ///////////////////////////////
static size_t MAXIMUM_STATIC_OVERLAPING_OPS_IN_SUBGRAPH = 3UL;
static size_t MAXIMUM_HEIGHT_WORTHY_FOR_VF = 38UL;
static size_t CMX_TO_AVOID_FRAGMENTATION = 360800;
static size_t YOLO_V4_NUMBER_OF_SUBS_DOUBLE_TAIL = 1;
static size_t YOLO_V4_NUMBER_OF_SUBS_SINGLE_TAIL = 11;
static size_t YOLO_V3_NUMBER_OF_SUBS_SINGLE_TAIL = 12;
////////////////////////////////////////////////////////////////////////////////

void populateCandidateVerticalFusionOps(std::vector<std::string> & candidateVerticalFusionOps, const std::vector<mv::Element> &strategyList,
    mv::OpModel &om, const uint64_t& cmxV)
{
    std::size_t nStreams = 0;
    std::unordered_map <std::string, uint8_t> type_size{{"Float16", 2}, {"UInt8", 1}};
    for (auto layerStrategy = strategyList.begin(); layerStrategy != strategyList.end(); ++layerStrategy)
    {
        auto layerNameStrategy = *layerStrategy;
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");
        bool isStreamingOnH = false;
        bool isStreamingOnlyOnH = false;
        for (auto splitNumber = splitList.begin(); splitNumber != splitList.end(); splitNumber++)
        {
            auto split = *splitNumber;
            std::string axis;
            if (split.attrsKeys().size() > 0)
                axis = split.attrsKeys()[0];
            else
                continue;
            auto numSplits = split.get<int>(axis);

            if (axis == "H")
            {
                if (numSplits > 1)
                {
                    isStreamingOnH = true;
                    nStreams = numSplits;
                }
            }
            else
                isStreamingOnlyOnH = (numSplits == 1);
        }
        auto op = om.getOp(nodeName);
        if (op->hasAttr("mixedToFloat") && op->get<bool>("mixedToFloat"))
            continue;
        if (op->isHardwarizable() && isStreamingOnH && isStreamingOnlyOnH)
        {
            auto outputTensor = op->getOutputTensor()[mv::IO_TENSOR_OUTPUT];
            auto inputTensor = op->getInputTensor()[mv::IO_TENSOR_OUTPUT];
            mv::Data::TensorIterator weightTensor;
            uint64_t weightResources = 0;
            if (op->hasWeights())
            {
                weightTensor = op->getInputTensor()[mv::IO_TENSOR_WEIGHTS_SET];
                weightResources = weightTensor->getShape().totalSize() *
                    type_size[weightTensor->get<mv::DType>("dType").toString()];
            }

            uint64_t outputResources = outputTensor->getShape().totalSize() *
                type_size[outputTensor->get<mv::DType>("dType").toString()];
            uint64_t inputResources = inputTensor->getShape().totalSize() *
                type_size[inputTensor->get<mv::DType>("dType").toString()]/nStreams;
            if (op->getOpType() == "Eltwise")
                weightResources = weightResources/nStreams;

            //NOTE: On YoloV3 there are cases that the algorithm will create subgraphs for the second
            //half of the network where we stream on small number of H just to be able to fit the input
            //tensor e.g. output Tensor dims = 512, 38, 38. Such a small number of lines with small number of
            //splits should not be attached to the subgraphs
            if (outputTensor->getShape()[mv::IO_HEIGHT_DIMENSION] <= MAXIMUM_HEIGHT_WORTHY_FOR_VF)
                continue;

            //NOTE: for now consider that an op is spilling if the output tensor is bigger than cmx
            if (inputResources + weightResources + outputResources > 0.8 * cmxV)
                candidateVerticalFusionOps.push_back(nodeName);
        }
    }
    return;
}

bool alreadyInSubgraph(const std::string &opName, const std::vector<std::list<std::string>> &candidateVerticalFusionOps)
{
    bool alreadyInSub = false;
    for (auto subgraphIt = candidateVerticalFusionOps.begin(); subgraphIt != candidateVerticalFusionOps.end(); ++subgraphIt)
    {
        if (std::find(subgraphIt->cbegin(), subgraphIt->cend(), opName) != subgraphIt->cend())
        {
            alreadyInSub = true;
            break;
        }
    }
    return alreadyInSub;
}

bool nodeIsNeighbour(mv::OpModel& om, mv::DataModel& dm, const std::string &opName, std::list<std::string>& subgraph)
{
    bool nodeIsNeighb = false;
    for (auto nodeName = subgraph.begin(); nodeName != subgraph.end(); ++nodeName)
    {
        auto opIt = om.getOp(*nodeName);
        std::vector<std::string> neighbors;
        mv::Data::OpListIterator previousOp;
        //NOTE: find Neighbors
        auto inputs = opIt->getInputTensor();
        // for (auto &inputTensor : opIt->getInputTensor())
        for (auto input = inputs.begin(); input != inputs.end(); ++input)
        {
            auto inputTensor = *input;
            if (!inputTensor->isPopulated())
                previousOp = om.getSourceOp(inputTensor);
            neighbors.push_back(previousOp->getName());
        }
        std::vector<mv::Data::OpListIterator> nextOps;
        // nextOps.reserve(mv::findSinkLayers(dm, opIt->getOutputTensor()[mv::IO_TENSOR_OUTPUT]).size());
        nextOps = mv::findSinkLayers(dm, opIt->getOutputTensor()[mv::IO_TENSOR_OUTPUT]);
        for (auto next = nextOps.begin(); next != nextOps.end(); ++next)
        {
            auto nextOp = *next;
            neighbors.push_back(nextOp->getName());
        }

        nodeIsNeighb = (std::find(neighbors.begin(), neighbors.end(), opName) != neighbors.end());
        if (nodeIsNeighb)
            break;
    }
    return nodeIsNeighb;
}

std::size_t computeMemoryResources(const mv::Data::OpListIterator& op, const std::size_t maxStream)
{
    std::unordered_map <std::string, uint8_t> type_size{{"Float16", 2}, {"UInt8", 1}};
    auto outputTensor = op->getOutputTensor()[0];
    std::size_t memoryResources = (outputTensor->getShape().totalSize() * type_size[outputTensor->getDType().toString()]/maxStream);
    for (std::size_t inputIdx = 0; inputIdx < op->getInputTensor().size(); ++inputIdx)
    {
        auto inputTensor = op->getInputTensor()[inputIdx];
        memoryResources += (inputTensor->getShape().totalSize() * type_size[inputTensor->getDType().toString()]/maxStream);
    }

    return memoryResources;
}

bool willMaxStreamingBePossible(mv::OpModel& om, const std::vector<mv::Element>& strategyList,
    const std::list<std::string>& subgraph, const std::string &opName)
{
    auto op = om.getOp(opName);
    std::set<int> streamNumbers = {};
    std::size_t maxStream = 1;
    for (auto layerStrategy = strategyList.begin(); layerStrategy != strategyList.end(); ++layerStrategy)
    {
        auto layerNameStrategy = *layerStrategy;
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");

        if (nodeName == opName || std::find(subgraph.begin(), subgraph.end(), nodeName) != subgraph.end())
            streamNumbers.insert(splitList[1].get<int>("H"));
    }

    std::size_t maxHeight = 1;
    for (auto& opIt : subgraph)
    {
        auto subbgraphOp = om.getOp(opIt);

        if (subbgraphOp->getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] > maxHeight)
            maxHeight = subbgraphOp->getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION];

        maxStream = *(streamNumbers.rbegin());
        if (maxStream > maxHeight)
            return false;
        //NOTE: try to balance the streams for vertical fusion
        while (maxHeight % maxStream >= maxHeight/maxStream)
            ++maxStream;

        while (computeMemoryResources(subbgraphOp, maxStream) > CMX_TO_AVOID_FRAGMENTATION)
            ++maxStream;

        streamNumbers.insert(maxStream);
        maxStream = *(streamNumbers.rbegin());

        auto inputHeight = op->getInputTensor(mv::IO_TENSOR_INPUT)->getShape()[mv::IO_HEIGHT_DIMENSION];
        if (maxStream > inputHeight)
            return false;

        std::size_t kernelHeight = 1;
        if (subbgraphOp->getOpType() == "Conv" || subbgraphOp->getOpType() == "DepthwiseConv")
        {
            auto weightTensorShape = subbgraphOp->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET)->getShape();
            kernelHeight = weightTensorShape[mv::IO_HEIGHT_DIMENSION];
        }
        // MaxPool kernel
        else if (subbgraphOp->hasAttr("kSize"))
        {
            auto kernel = subbgraphOp->get<std::array<unsigned short, 2UL>>("kSize");
            kernelHeight = kernel[mv::IO_HEIGHT_DIMENSION];
        }
        if (kernelHeight > inputHeight/maxStream)
            return false;
    }

    return true;
}

bool hasKernelNotEqualStride(mv::Data::OpListIterator testOp)
{
    // static rule good to have for accuracy issues
    // cost model should decide
    bool hasKernelNotEqualStrideFlag = false;
    if (testOp->getOpType() == "Conv" || testOp->getOpType() == "DepthwiseConv")
    {
        auto weightTensorShape = testOp->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET)->getShape();
        auto kernelHeight = weightTensorShape[mv::IO_HEIGHT_DIMENSION];
        if (testOp->hasAttr("stride"))
        {
            auto strideHeight = testOp->get<std::array<unsigned short, 2>>("stride")[mv::IO_HEIGHT_DIMENSION];
            if (kernelHeight != strideHeight)
                hasKernelNotEqualStrideFlag = true;
        }
        else
            hasKernelNotEqualStrideFlag = true;
    }
    // MaxPool kernel
    else if (testOp->hasAttr("kSize"))
    {
        auto kernel = testOp->get<std::array<unsigned short, 2UL>>("kSize");
        auto kernelHeight = kernel[mv::IO_HEIGHT_DIMENSION];
        if (testOp->hasAttr("stride"))
        {
            auto strideHeight = testOp->get<std::array<unsigned short, 2>>("stride")[mv::IO_HEIGHT_DIMENSION];
            if (kernelHeight != strideHeight)
                hasKernelNotEqualStrideFlag = true;
        }
        else
            hasKernelNotEqualStrideFlag = true;
    }
    return hasKernelNotEqualStrideFlag;
}

bool kernelLargeOverlap(mv::OpModel& om, const std::list<std::string>& subgraph, const std::string &opName)
{
    // only allow 3 convolution with kernel != 1x1 in one subgraph
    auto testOp = om.getOp(opName);
    size_t count = 1;
    if (hasKernelNotEqualStride(testOp))
    {
        for (auto& op : subgraph)
        {
            auto subbgraphOp = om.getOp(op);
            if (hasKernelNotEqualStride(subbgraphOp))
                ++count;
            if (count > MAXIMUM_STATIC_OVERLAPING_OPS_IN_SUBGRAPH) { return true; }
        }
    }
    return false;
}

bool majorityOpsWithLargeKernel(mv::OpModel& om, const std::list<std::string>& subgraph)
{
    // if half or more of the ops have large kernel, discard that subgraph
    size_t count = 0;
    for (auto& op : subgraph)
    {
        auto subbgraphOp = om.getOp(op);
        if (hasKernelNotEqualStride(subbgraphOp))
            ++count;
    }
    return (count > subgraph.size()/2);
}

bool tailInputOpsOutsideSubgraph(mv::OpModel& om, const std::list<std::string>& subgraph)
{
    // subgraph tail operation which is a concat or an eltwise must have all
    // output flows of the input ops in the subgraph
    // due to the tiling and streaming calculations
    for (auto& opName : subgraph)
    {
        auto op = om.getOp(opName);
        if (op->getOpType() == "Concat" || op->getOpType() == "Eltwise")
        {
            for (auto& inTensor : op->getInputTensor())
            {
                if (!inTensor->isPopulated())
                {
                    auto inputOp = om.getSourceOp(inTensor);
                    if (std::find(subgraph.begin(), subgraph.end(), inputOp->getName()) == subgraph.end())
                    {
                        // if input op is not in the subgraph assert that all children are
                        // if the tail is the only child
                        if (inputOp.childrenSize() == 1)
                            return true;
                        auto childItr = inputOp.leftmostChild();
                        // else verify that all childern are in the subgraph
                        for ( ; childItr != om.opEnd(); ++childItr)
                            if (std::find(subgraph.begin(), subgraph.end(), childItr->getName()) == subgraph.end())
                                return true;
                    }
                }
            }
        }
    }
    return false;
}

bool opHasChildernOutsideSubgraph(mv::OpModel& om, mv::Data::OpListIterator& currOp,
    const std::string& tailName, const std::list<std::string>& subgraph)
{
    if (currOp->getName() != tailName)
        for (auto childItr = currOp.leftmostChild(); childItr != om.opEnd(); ++childItr)
            if (std::find(subgraph.begin(), subgraph.end(), childItr->getName()) == subgraph.end())
                return true;
    return false;
}

bool opHasParentsOutsideSubgraph(mv::OpModel& om, mv::Data::OpListIterator& currOp,
    const std::string& headName, const std::list<std::string>& subgraph)
{
    if (currOp->getName() != headName)
        for (auto inTensor : currOp->getInputTensor())
            if (!inTensor->isPopulated())
                if (std::find(subgraph.begin(), subgraph.end(), om.getSourceOp(inTensor)->getName()) == subgraph.end())
                    return true;
    return false;
}

bool externalOpsRequireSubgraphOps(mv::OpModel& om, const std::list<std::string>& subgraph)
{
    // the subgraph should be closed from the head op to the tail op
    // ensure all intermediate ops between head and tail are in the subgraph
    auto headName = subgraph.front();
    auto tailName = subgraph.back();
    for (auto& opName : subgraph)
    {
        auto currOp = om.getOp(opName);
        if (opHasChildernOutsideSubgraph(om, currOp, tailName, subgraph))
            return true;
        // NOTE: tail condition checked in tailInputOpsOutsideSubgraph()
        if (opName != tailName && opHasParentsOutsideSubgraph(om, currOp, headName, subgraph))
            return true;
    }

    return false;
}

bool isNotValidSubgraph(mv::OpModel& om, const std::list<std::string>& subgraph)
{
    // remove any complex/invalid subgraphs
    if (majorityOpsWithLargeKernel(om, subgraph))
        return true;
    else if (tailInputOpsOutsideSubgraph(om, subgraph))
        return true;
    else if (externalOpsRequireSubgraphOps(om, subgraph))
        return true;
    else
        return false;
}

void sortCandidateOps(mv::OpModel& om, std::vector<std::string>& candidateVerticalFusionOpsSorted, std::vector<std::string>& candidateVerticalFusionOps)
{
    auto sortedOps = om.topologicalSort();
    for (auto sortedOp = sortedOps.begin(); sortedOp != sortedOps.end(); ++sortedOp)
    {
        auto sortOp = *sortedOp;
        auto exists = std::find(candidateVerticalFusionOps.begin(), candidateVerticalFusionOps.end(), sortOp->getName()) != candidateVerticalFusionOps.end();
        if (exists)
            candidateVerticalFusionOpsSorted.push_back(sortOp->getName());
    }
    return;
}

bool childConcat(mv::DataModel& dm, mv::OpModel& om, const std::string &opName)
{
    bool concatFound = false;
    auto nextOps = mv::findSinkLayers(dm, om.getOp(opName)->getOutputTensor()[0]);
    for (auto next = nextOps.begin(); next != nextOps.end(); ++next)
    {
        auto nextOp = *next;
        if (nextOp->getOpType() == "Concat" || nextOp->getOpType() == "ImplicitConcat")
        {
            concatFound = true;
            break;
        }
    }
    return concatFound;
}

static void saveNewStreamingStrategiesToJson(const mv::Attribute& streamingStrategyElements)
{
    std::ofstream jsonOutputFile;
    std::string jsonOutFileName = "./output/vertical_fusion.json";
    jsonOutputFile.open(jsonOutFileName, std::ios::out);

    mv::Element SSA("Streaming strategies for vertical fusion generated by mcmCompiler ");
    SSA.set("streaming_strategy", streamingStrategyElements);
    auto jsonSStrategy = SSA.toJSON(true);

    jsonOutputFile << jsonSStrategy.stringifyPretty() << "," << std::endl;
    jsonOutputFile.close();
}

bool memberOfSubgraph(const mv::Data::OpListIterator &op)
{
    bool memberOfSubgraph = ((op->hasAttr("verticalFusionSubgraphHead") && op->get<bool>("verticalFusionSubgraphHead")) ||
            (op->hasAttr("verticalFusion") && op->get<bool>("verticalFusion")) ||
            (op->hasAttr("verticalFusionSubgraphTail") && op->get<bool>("verticalFusionSubgraphTail")));
    return memberOfSubgraph;
}

void printVerticalFusionSubgraphs(const std::vector<std::list<std::string>> &verticalFusionSubgraphs, const std::vector<mv::Element>& newStreamingStrategies)
{
    std::size_t streamsOnH = 0, idx = 0;
    //NOTE: functionality that prints the subgraphs and the strategies that they are assigned with
    for (auto subgraph = verticalFusionSubgraphs.begin(); subgraph != verticalFusionSubgraphs.end(); subgraph++)
    {
        std::cout << "Printing info for the subgraph " << idx << std::endl;
        for (auto opIt = subgraph->begin(); opIt != subgraph->end(); ++opIt)
        {
            for (auto layerStrategy = newStreamingStrategies.begin(); layerStrategy != newStreamingStrategies.end(); ++layerStrategy)
            {
                auto layerNameStrategy = *layerStrategy;
                std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
                auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                if (nodeName == *opIt)
                {
                    streamsOnH = splitList[1].get<int>("H");
                    std::cout << nodeName << ": " << streamsOnH << std::endl;
                    break;
                }
            }
        }
        idx++;
    }
    return;
}

void storeOverlappingEltwiseLines(const std::vector<std::list<std::string>> &verticalFusionSubgraphs, mv::OpModel& om)
{
    for (auto subgraph = verticalFusionSubgraphs.begin(); subgraph != verticalFusionSubgraphs.end(); subgraph++)
    {
        std::vector<mv::Data::OpListIterator> eltwises = {};
        std::vector<mv::Data::OpListIterator> tails = {};
        for (auto opIt = subgraph->begin(); opIt != subgraph->end(); ++opIt)
        {
            auto op = om.getOp(*opIt);
            if (op->getOpType() == "Eltwise")
                eltwises.push_back(op);
            else if (op->hasAttr("verticalFusionSubgraphTail") && op->get<bool>("verticalFusionSubgraphTail"))
                tails.push_back(op);
        }

        for (auto eltwiseOp = eltwises.begin(); eltwiseOp != eltwises.end(); eltwiseOp++)
        {
            std::size_t overLappingSubgraphOpsIndex = 0;
            for (auto tailOp = tails.begin(); tailOp != tails.end(); ++tailOp)
            {
                if (!om.pathExists(*eltwiseOp, *tailOp))
                {
                    continue;
                }
                else
                {
                    auto tempOp = om.getOp((*eltwiseOp)->getName());
                    while (tempOp->getName() != (*tailOp)->getName())
                    {
                        if (hasKernelNotEqualStride(tempOp))
                            ++overLappingSubgraphOpsIndex;
                        ++tempOp;
                    }
                    auto eltwise = *eltwiseOp;
                    eltwise->set<std::size_t>("overLappingSubgraphOpsIndex", overLappingSubgraphOpsIndex);
                }
            }
        }
    }
    return;
}

bool after_concat(const std::string &name, mv::OpModel& om)
{
    auto previousOp = om.getSourceOp(om.getOp(name)->getInputTensor()[0]);
    return (previousOp->getOpType() == "Concat" ||  previousOp->getOpType() == "ImplicitConcat");
}

bool excludedFromSubgraphs(const std::string &name, const std::vector<std::string> excluded)
{
    auto exists = std::find(excluded.begin(), excluded.end(), name) != excluded.end();
    return exists;
}

bool subgraphHasAnOpFollowedFromConcat(const std::list<std::string> sugraphOps, const std::set<std::string> followingConcats)
{
    bool subgraphHasAnOpFollowedFromConcat = false;
    for (auto opName = sugraphOps.begin(); opName != sugraphOps.end(); ++opName)
    {
        auto exists = std::find(followingConcats.begin(), followingConcats.end(), *opName) != followingConcats.end();
        if (exists)
        {
            subgraphHasAnOpFollowedFromConcat = true;
            break;
        }
    }
    return subgraphHasAnOpFollowedFromConcat;
}

void computeSubgraphs(mv::ComputationModel& model,
                                mv::Element& passDesc,
                                bool double_tail)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    //step-0: read Arguments
    std::vector<std::string> candidateVerticalFusionOps;
    auto globalParams = model.getGlobalConfigParams();
    bool vertical_fusion = passDesc.hasAttr("vertical_fusion") ? passDesc.get<bool>("vertical_fusion"): false;
    if (!vertical_fusion)
        return;
    int cmx = model.getGlobalConfigParam("cmx").get<int>();
    if (cmx <= 0)
        throw mv::RuntimeError("VerticalFusion", "cmx value is not well defined");
    const uint64_t cmxV = cmx;

    if (!globalParams->hasAttr("streaming_strategy"))
        return;

    int min_subgraph_depth = passDesc.hasAttr("min_subgraph_depth") ? passDesc.get<int>("min_subgraph_depth"): 2;

    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    //step-1: read the strategies and populate a map with the candidate vertical fusion ops, whis means all the ops
    //that are streaming on H and are spilling back to ddr
    populateCandidateVerticalFusionOps(candidateVerticalFusionOps, strategyList, om, cmxV);

    //step-2: sort the candidate ops following the topoligal order
    std::vector<std::string> candidateVerticalFusionOpsSorted;
    sortCandidateOps(om, candidateVerticalFusionOpsSorted, candidateVerticalFusionOps);

    std::set<std::string> followedFromConcatOps;
    //find which ops are followed by a concat
    for (auto opName = candidateVerticalFusionOpsSorted.begin(); opName != candidateVerticalFusionOpsSorted.end(); ++opName)
    {
        //NOTE: this is a specific modification for the yolo subgraphs
        bool concatFound = childConcat(dm, om, *opName);
        if (concatFound)
            followedFromConcatOps.insert(*opName);
    }

    if (!double_tail)
    {
        //step-3: if a candidate has concat as childs, remove
        //NOTE: this is a specific modification for the yolo subgraphs
        auto opName = candidateVerticalFusionOpsSorted.begin();
        while (opName != candidateVerticalFusionOpsSorted.end())
        {
            //NOTE: this is a specific modification for the yolo subgraphs
            bool concatExists = (std::find(followedFromConcatOps.begin(), followedFromConcatOps.end(), *opName) != followedFromConcatOps.end());
            if (concatExists)
                candidateVerticalFusionOpsSorted.erase(opName);
            else
                opName++;
        }

        //step-4: if a candidate has multiple sinks, if not everybody is a candidate then remove the first candidate
        auto sortedName = candidateVerticalFusionOpsSorted.begin();
        while (sortedName != candidateVerticalFusionOpsSorted.end())
        {
            auto sortedOp = om.getOp(*sortedName);
            auto sinkOps = mv::findSinkLayers(dm, sortedOp->getOutputTensor()[0]);
            bool sinkIsCandidate = true;
            bool erasedName = false;
            if (sinkOps.size() > 1)
            {
                for (auto sink = sinkOps.begin(); sink != sinkOps.end(); ++sink)
                {
                    sinkIsCandidate = std::find(candidateVerticalFusionOpsSorted.begin(), candidateVerticalFusionOpsSorted.end(), (*sink)->getName()) != candidateVerticalFusionOpsSorted.end();
                    if (!sinkIsCandidate)
                    {
                        auto itr = std::find(candidateVerticalFusionOpsSorted.begin(), candidateVerticalFusionOpsSorted.end(), *sortedName);
                        if (itr != candidateVerticalFusionOpsSorted.end())
                        {
                            candidateVerticalFusionOpsSorted.erase(itr);
                            erasedName = true;
                            break;
                        }
                    }
                }
            }
            if (!erasedName)
                sortedName++;
        }
    }

    //step-4: for every op go and locate the vertical fusion subgraphs
    std::vector<std::list<std::string>> verticalFusionSubgraphs;
    std::vector<std::string> excludedVFnodes;
    std::size_t numberOfSubgraphs = 0;

    for (auto name = candidateVerticalFusionOpsSorted.begin(); name != candidateVerticalFusionOpsSorted.end(); ++name)
    {
        if (alreadyInSubgraph(*name, verticalFusionSubgraphs))
            continue;
        if (double_tail)
        {
            //NOTE: jump the huge operation after first concat subgraph
            if (after_concat(*name, om))
            {
                excludedVFnodes.push_back(*name);
                continue;
            }
        }
        auto root_subgraph = *name;
        verticalFusionSubgraphs.push_back({root_subgraph});

        for (auto node = candidateVerticalFusionOpsSorted.begin(); node != candidateVerticalFusionOpsSorted.end(); ++node)
        {
            if (alreadyInSubgraph(*node, verticalFusionSubgraphs) || (excludedFromSubgraphs(*node, excludedVFnodes)))
                continue;

            //NOTE: the target here is to find out that all the ops that belong to the same subgraph will be
            //divisible with the maximum number of streams
            bool streamPossible = willMaxStreamingBePossible(om, strategyList, verticalFusionSubgraphs[numberOfSubgraphs], *node);

            bool isNeighbour = nodeIsNeighbour(om, dm, *node, verticalFusionSubgraphs[numberOfSubgraphs]);
            bool kernelLargeOverlapFlag = kernelLargeOverlap(om, verticalFusionSubgraphs[numberOfSubgraphs], *node);
            if (isNeighbour && streamPossible && !kernelLargeOverlapFlag)
            {
                //step-4.1: special for yolo architectures, prune to eltwise as last node
                verticalFusionSubgraphs[numberOfSubgraphs].push_back(*node);
                if (double_tail)
                {
                    if (om.getOp(*node)->getOpType() == "Concat")
                        break;
                }
                else
                {
                    if (om.getOp(*node)->getOpType() == "Eltwise")
                        break;
                }
            }
        }
        if (verticalFusionSubgraphs[numberOfSubgraphs].size() <= std::size_t(min_subgraph_depth))
            verticalFusionSubgraphs.erase(verticalFusionSubgraphs.begin() + numberOfSubgraphs);
        //NOTE: normally this condition should check if the tail kernel != stride and the tail that is later than the head
        else if (!double_tail && numberOfSubgraphs == YOLO_V4_NUMBER_OF_SUBS_SINGLE_TAIL)
            verticalFusionSubgraphs.erase(verticalFusionSubgraphs.begin() + numberOfSubgraphs);
        else if (double_tail && numberOfSubgraphs >= YOLO_V4_NUMBER_OF_SUBS_DOUBLE_TAIL)
            verticalFusionSubgraphs.erase(verticalFusionSubgraphs.begin() + numberOfSubgraphs);
        else if (!double_tail && isNotValidSubgraph(om, verticalFusionSubgraphs[numberOfSubgraphs]))
            verticalFusionSubgraphs.erase(verticalFusionSubgraphs.begin() + numberOfSubgraphs);
        else
            numberOfSubgraphs++;
    }

    //NOTE: this idx value will be used for hardcoding the vertical fusion tiling size in order to avoid overfitting cmx
    //normally here we should have a cost function that will predict the appropriate number of tiles
    std::size_t idx = 0;
    std::vector<std::size_t> verticalTiles;
    bool yolov4 = false;
    if (verticalFusionSubgraphs.size() == YOLO_V3_NUMBER_OF_SUBS_SINGLE_TAIL || verticalFusionSubgraphs.size() == YOLO_V4_NUMBER_OF_SUBS_SINGLE_TAIL ||
        (double_tail && verticalFusionSubgraphs.size() == YOLO_V4_NUMBER_OF_SUBS_DOUBLE_TAIL))
    {
        yolov4 = true;
        om.getInput()->set<bool>("yolo_v4", true);
    }

    if (yolov4 && !double_tail)
        verticalTiles = {38, 19, 19, 8, 4, 4, 4, 4, 4, 4, 4, 6};
    else if (yolov4 && double_tail)
        verticalTiles = {38, 19};

    std::vector<mv::Element> newStreamingStrategies;
    //NOTE: store attributes so later the streaming pass can handle the subgraphs appropriately
    for (auto subgraph = verticalFusionSubgraphs.begin(); subgraph != verticalFusionSubgraphs.end(); subgraph++)
    {
        auto head = subgraph->front();
        auto tail = subgraph->back();
        for (auto it = subgraph->begin(); it != subgraph->end(); ++it)
            om.getOp(*it)->set<bool>("verticalFusion", true);

        om.getOp(head)->set<bool>("verticalFusionSubgraphHead", true);
        om.getOp(tail)->set<bool>("verticalFusionSubgraphTail", true);
        om.getOp(head)->set<bool>("verticalFusion", false);
        om.getOp(tail)->set<bool>("verticalFusion", false);

        if (double_tail)
        {
            bool subgraphHasAnOpFollowedConcat = subgraphHasAnOpFollowedFromConcat(*subgraph, followedFromConcatOps);
            for (auto it = subgraph->begin(); it != subgraph->end(); ++it)
            {
                bool concatExists = (std::find(followedFromConcatOps.begin(), followedFromConcatOps.end(), *it) != followedFromConcatOps.end());
                if (concatExists)
                {
                    om.getOp(*it)->set<bool>("verticalFusionSubgraphTail", true);
                    om.getOp(*it)->set<bool>("verticalFusion", false);
                }
                //remove previous tails
                else if (subgraphHasAnOpFollowedConcat)
                    om.getOp(*it)->set<bool>("verticalFusionSubgraphTail", false);

                if (*it != head)
                    om.getOp(*it)->set<bool>("verticalFusionSubgraphHead", false);
            }
        }

        std::set<int> streamNumbers = {};
        int maxStream = 0;
        for (auto it = subgraph->begin(); it != subgraph->end(); ++it)
        {
            for (auto layerStrategy = strategyList.begin(); layerStrategy != strategyList.end(); ++layerStrategy)
            {
                auto layerNameStrategy = *layerStrategy;
                std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
                auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                if (nodeName == *it)
                    streamNumbers.insert(splitList[1].get<int>("H"));
            }
        }

        std::size_t maxHeight = 1;
        for (auto it = subgraph->begin(); it != subgraph->end(); ++it)
        {
            auto op = om.getOp(*it);
            if (op->getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] > maxHeight)
                maxHeight = op->getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION];
        }

        for (auto it = subgraph->begin(); it != subgraph->end(); ++it)
        {
            maxStream = *(streamNumbers.rbegin());
            //NOTE: try to balance the streams for vertical fusion
            while (maxHeight % maxStream >= maxHeight/maxStream)
                ++maxStream;

            //NOTE: try to avoid fragmentation of lpscheduler providing dpu tasks with less than 350K
            auto op = om.getOp(*it);

            while (computeMemoryResources(op, maxStream) > CMX_TO_AVOID_FRAGMENTATION)
                ++maxStream;

            streamNumbers.insert(maxStream);
        }

        for (auto it = subgraph->begin(); it != subgraph->end(); ++it)
        {
            if (!streamNumbers.empty())
            {
                if (yolov4)
                    maxStream = verticalTiles[idx];
                else
                    maxStream = *(streamNumbers.rbegin());
            }

            for (auto layerStrategy = strategyList.begin(); layerStrategy != strategyList.end(); ++layerStrategy)
            {
                auto layerNameStrategy = *layerStrategy;
                std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
                auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                if (nodeName == *it)
                {
                    splitList[1].get<int>("H") = maxStream;

                    mv::Element element("");
                    element.set("name_filter", nodeName);
                    std::vector<mv::Element> copySplits(splitList.size(), mv::Element(""));
                    copySplits[0].set<int>("W", 1);
                    copySplits[1].set<int>("H", maxStream);
                    copySplits[2].set<int>("C", 1);
                    copySplits[3].set<int>("K", 1);
                    copySplits[4].set<int>("N", 1);

                    element.set("splits", copySplits);
                    newStreamingStrategies.emplace_back(std::move(element));
                }
            }
        }
        idx = idx + 1;
    }
    //NOTE - now attach the layer that do not belong to a subgraph
    for (auto layerStrategy = strategyList.begin(); layerStrategy != strategyList.end(); ++layerStrategy)
    {
        auto layerNameStrategy = *layerStrategy;
        bool exists = false;
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        for (auto subgraph = verticalFusionSubgraphs.begin(); subgraph != verticalFusionSubgraphs.end(); ++subgraph)
        {
            if(std::find(subgraph->cbegin(), subgraph->cend(), nodeName) != subgraph->cend())
            {
                exists = true;
                break;
            }
        }
        if (!exists)
            newStreamingStrategies.emplace_back(std::move(layerNameStrategy));

        globalParams->set<std::vector<mv::Element>>("streaming_strategy", newStreamingStrategies);
        saveNewStreamingStrategiesToJson(newStreamingStrategies);
    }

    // printVerticalFusionSubgraphs(verticalFusionSubgraphs, newStreamingStrategies);
    if (double_tail)
        storeOverlappingEltwiseLines(verticalFusionSubgraphs, om);

    return;
}

void printSubgraphsAfterComputation(mv::ComputationModel& model)
{
    mv::OpModel om(model);
    auto sortedOps = om.topologicalSort();
    for (auto opIt = sortedOps.begin(); opIt != sortedOps.end(); ++opIt)
    {
        auto op = *opIt;
        if (op->hasAttr("verticalFusionSubgraphHead") && op->get<bool>("verticalFusionSubgraphHead"))
            std::cout << "Head of subgraph with name " << op->getName() << std::endl;
        if (op->hasAttr("verticalFusion") && op->get<bool>("verticalFusion"))
            std::cout << "Member of subgraph with name " << op->getName() << std::endl;
        if (op->hasAttr("verticalFusionSubgraphTail") && op->get<bool>("verticalFusionSubgraphTail"))
            std::cout << "Tail of subgraph with name " << op->getName() << std::endl;
    }
    return;
}

void recognizeVerticalFusionPatternsFcn(const mv::pass::PassEntry&,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element& passDesc,
                                mv::Element&)
{
    //NOTE: the following pass will compute the operations that belong to vertical fusion subgraphs.
    //There are 2 kind of subgraphs the ones that will have double tail, which means that the subgraph
    //finishes just before a concat with 2 inputs, like is happenning in yolov4 and the single tail
    //which finishes with an eltwise add operation. The pass takes all the operations that are
    //streaming on height dimension and are spilling, sorts the ops and adds them recursively till
    //it generates the subgraphs the way that vertical fusion is waiting them.
    //https://docs.intel.com/documents/MovidiusInternal/vpu27/common/SW/HLD/02_02_NN_CompilerHLD.html#vertical-fusion
    //The idea is that you generate the subgraphs that are expected to have single tail, if the architecture
    //supports double tail as well, re-compute and then fuse
    bool vertical_fusion = passDesc.hasAttr("vertical_fusion") ? passDesc.get<bool>("vertical_fusion"): false;
    if (!vertical_fusion)
        return;

    mv::OpModel om(model);
    computeSubgraphs(model, passDesc, false);
    bool yolo_v4 = om.getInput()->hasAttr("yolo_v4") && om.getInput()->get<bool>("yolo_v4");
    if (yolo_v4)
        computeSubgraphs(model, passDesc, yolo_v4);
    // printSubgraphsAfterComputation(model);
    return;
}

void verticalFusionTransformationFcn(const mv::pass::PassEntry&,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element& ,
                                mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    //NOTE: The target of this pass is to find and remove the ops that have been inserted from streaming
    //in the vertical fusion subgraphs from streaming and should not be there
    auto sortedOps = om.topologicalSort();
    mv::Data::OpListIterator concatToRemove;
    std::string previousParent = "";
    std::string currentParent;

    //step-0: iterate to the vertical fusion subgraphs and remove all the concats apart from the last one.
    auto opIt = sortedOps.begin();
    while (opIt != sortedOps.end())
    {
        auto op = *opIt;
        if (op->getOpType() == "Output")
            break;

        if (op->hasAttr("parentOpName"))
            currentParent = op->get<std::string>("parentOpName");

        if (previousParent != "" && previousParent != currentParent)
        {
            if (!(concatToRemove->hasAttr("concatTail") && concatToRemove->get<bool>("concatTail")) &&
                memberOfSubgraph(op))
                om.removeOp(concatToRemove);
        }
        auto concatOps = mv::findSinkLayers(dm, op->getOutputTensor()[0]);
        std::vector<mv::Data::OpListIterator> nextOps;

        //NOTE: throw in case of not 1 nextOp
        if ((op->hasAttr("verticalFusionSubgraphHead") && op->get<bool>("verticalFusionSubgraphHead")) ||
            (op->hasAttr("verticalFusion") && op->get<bool>("verticalFusion")))
        {
            if (concatOps.empty())
                throw mv::RuntimeError("Vertical Fusion", "Incorrect subgraph arrived for vertical fusion");

            auto outputTensor = op->getOutputTensor()[0];
            if (concatOps[0]->getOpType() == "Concat" || concatOps[0]->getOpType() == "ImplicitConcat")
                nextOps = mv::findSinkLayers(dm, concatOps[0]->getOutputTensor()[0]);
            else if (concatOps[0]->getOpType() == "Slice")
            {
                opIt++;
                continue;
            }
            else
                throw mv::RuntimeError("Vertical Fusion", "Incorrect subgraph arrived for vertical fusion");

            //NOTE: throw in case of inputs of concat != concat
            auto parOp = concatOps[0].leftmostParent();
            auto nextOp = concatOps[0].leftmostChild();
            std::size_t parents = 0, childs = 0;
            while(parOp != om.opEnd())
            {
                ++parents;
                ++parOp;
            }
            while(nextOp != om.opEnd())
            {
                ++childs;
                ++nextOp;
            }
            //NOTE: in a subgraph the flows of input will be equal with the flows of output, unless
            //an eltwise will be a member of the subgraph were we can have double relationship
            if ((childs != parents) && (childs != 2*parents))
                throw mv::RuntimeError("Vertical Fusion", "Concat member of a vertical fusion subgraph " + concatOps[0]->getName()
                    + " should have number of inputs equal or double*2 of outputs but has "
                    + std::to_string(parents) + " " + std::to_string(childs));

            std::size_t iterations = 1;
            if (childs == 2 * parents)
                iterations = 2;

            auto nextOperation = concatOps[0].leftmostChild();
            for (std::size_t it = 0; it < iterations; ++it)
            {
                auto parentOperation = concatOps[0].leftmostParent();
                while (parentOperation != om.opEnd())
                {
                    if (!om.edgeExists(parentOperation, nextOperation))
                    {
                        nextOperation->setInputTensor(parentOperation->getOutputTensor()[0], 0, false);
                        om.defineFlow(parentOperation->getOutputTensor()[0], nextOperation, 0);
                    }
                    ++parentOperation;
                    ++nextOperation;
                }
            }
        }
        if  (op->hasAttr("parentOpName") &&
            memberOfSubgraph(op))
        {
            previousParent = op->get<std::string>("parentOpName");
            concatToRemove = concatOps[0];
        }
        opIt++;
    }
    //step-1: Iterate all over the ops of vertical fusion and remove the slice operations from the subgraph.
    sortedOps = om.topologicalSort();
    opIt = sortedOps.begin();
    while (opIt != sortedOps.end())
    {
        auto op = *opIt;
        if ((op->hasAttr("verticalFusionSubgraphTail") && op->get<bool>("verticalFusionSubgraphTail")) ||
            (op->hasAttr("verticalFusion") && op->get<bool>("verticalFusion")))
        {
            auto outputTensor = op->getOutputTensor()[mv::IO_TENSOR_OUTPUT];
            for (std::size_t idx = 0; idx < op->getInputTensor().size(); ++idx)
            {
                auto inputTensor = op->getInputTensor()[idx];
                if (!inputTensor->isPopulated())
                {
                    auto previousOp = om.getSourceOp(inputTensor);
                    if (previousOp->getOpType() != "Slice")
                        throw mv::RuntimeError("Vertical Fusion", "Incorrect subgraph arrived for vertical fusion, need to delete slice");

                    auto parentOp = om.getSourceOp(previousOp->getInputTensor()[mv::IO_TENSOR_OUTPUT]);
                    if (parentOp->hasAttr("concatTail") && parentOp->get<bool>("concatTail") &&
                        op->hasAttr("verticalFusionSubgraphTail") && op->get<bool>("verticalFusionSubgraphTail"))
                        continue;
                    else
                    {
                        if (!om.edgeExists(parentOp, op))
                        {
                            auto sourceTensor = parentOp->getOutputTensor()[mv::IO_TENSOR_OUTPUT];
                            op->setInputTensor(sourceTensor, idx, false);
                            om.defineFlow(sourceTensor, op, idx);
                            om.removeOp(previousOp);
                        }
                    }
                }
            }
        }
        opIt++;
    }
    return;
}

void validateVerticalAdds(const mv::pass::PassEntry&,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element&,
                                mv::Element&)
{
    //NOTE: it is possible that we will need to balance the eltwise input tiles, cause in some
    //vertical fusion subgraphs they might not agree, e.g. a->b->c->add, a->add, nobody can
    //guarantee that the height of the output tensor of a and c, which are the inputs of eltwise
    //will agree, have same height.
    mv::OpModel om(model);
    mv::DataModel dm(model);
    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto eltwOp = *opIt;
        auto op = *opIt;
        if (eltwOp.getOpType() == "DPUTask"
            && eltwOp.get<std::string>("taskOp") == "Eltwise")
        {
            auto outputTensor = eltwOp.getOutputTensor()[0];
            std::size_t idx = 0;
            mv::Data::TensorIterator slice;
            mv::Data::FlowListIterator outputFlow;
            bool flowToDelete = false;

            if (eltwOp.getInputTensor()[1]->getShape()[mv::IO_HEIGHT_DIMENSION] >
                eltwOp.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION])
            {
                auto previousOp = om.getSourceOp(eltwOp.getInputTensor()[1]);
                outputFlow = previousOp.leftmostOutput();
                while (outputFlow != om.flowEnd())
                {
                    if (outputFlow.sink()->getName() == eltwOp.getName())
                        break;
                    ++outputFlow;
                }
                auto initialOpId = previousOp->get<unsigned>("opId");
                std::size_t startingPoint = 0;
                if (previousOp->hasAttr("doubleTensorOverlappingLines"))
                    startingPoint = previousOp->get<std::size_t>("doubleTensorOverlappingLines");

                slice = om.slice("Slice" + previousOp->getName(), eltwOp.getInputTensor()[1], {0, startingPoint,0,0},
                    eltwOp.getInputTensor()[0]->getShape());
                slice->setQuantParams(eltwOp.getInputTensor()[1]->getQuantParams());
                auto sliceOp = om.getSourceOp(slice);
                sliceOp->set<unsigned>("opId", initialOpId);
                idx = 1;
                eltwOp.getInputTensor()[1]->set<bool>("doubleTest", true);
                flowToDelete = true;
            }
            else if (eltwOp.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] >
                eltwOp.getInputTensor()[1]->getShape()[mv::IO_HEIGHT_DIMENSION])
            {
                auto previousOp = om.getSourceOp(eltwOp.getInputTensor()[0]);
                outputFlow = previousOp.leftmostOutput();
                while (outputFlow != om.flowEnd())
                {
                    if (outputFlow.sink()->getName() == eltwOp.getName())
                        break;
                    ++outputFlow;
                }
                auto initialOpId = previousOp->get<unsigned>("opId");
                std::size_t startingPoint = 0;
                if (previousOp->hasAttr("doubleTensorOverlappingLines"))
                    startingPoint = previousOp->get<std::size_t>("doubleTensorOverlappingLines");

                slice = om.slice("Slice" + previousOp->getName(), eltwOp.getInputTensor()[0], {0,startingPoint,0,0},
                    eltwOp.getInputTensor()[1]->getShape());
                slice->setQuantParams(eltwOp.getInputTensor()[0]->getQuantParams());
                auto sliceOp = om.getSourceOp(slice);
                sliceOp->set<unsigned>("opId", initialOpId);
                idx = 0;
                eltwOp.getInputTensor()[0]->set<bool>("doubleTest", true);
                flowToDelete = true;
            }
            if (flowToDelete)
            {
                slice->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);
                om.undefineFlow(outputFlow);
                opIt->setInputTensor(slice, idx, false);
                om.defineFlow(slice, opIt, idx);
            }
        }
        if (op.hasAttr("vertical_fusion_overlap")  && op.get<bool>("vertical_fusion_overlap"))
        {
            auto outputTensor = op.getOutputTensor()[0];
            std::size_t idx = 0;
            mv::Data::TensorIterator slice;
            mv::Data::FlowListIterator inputFlow;

            auto nextOp = mv::findSinkLayers(dm, op.getOutputTensor()[0])[0];
            inputFlow = nextOp.leftmostOutput();
            while (inputFlow != om.flowEnd())
            {
                if (inputFlow.source()->getName() == op.getName())
                    break;
                ++inputFlow;
            }
            for (std::size_t i = 0u; i < nextOp->getInputTensor().size(); ++i)
            {
                if (nextOp->getInputTensor()[i]->getName() == op.getOutputTensor()[0]->getName())
                {
                    idx = i;
                    break;
                }
            }
            auto initialOpId = nextOp->get<unsigned>("opId");
            std::size_t startingPoint = 0;
            if (op.hasAttr("concatOverLappingLines"))
                startingPoint = op.get<std::size_t>("concatOverLappingLines");

            auto initialShape = op.getOutputTensor()[0]->getShape();
            auto slicedShape = mv::Shape({initialShape[mv::IO_WIDTH_DIMENSION], initialShape[mv::IO_HEIGHT_DIMENSION] - startingPoint, initialShape[mv::IO_CHANNEL_DIMENSION], 1});
            slice = om.slice("Slice" + op.getName(), op.getOutputTensor()[0], {0, startingPoint,0,0},
                slicedShape);
            slice->setQuantParams(op.getOutputTensor()[0]->getQuantParams());
            auto sliceOp = om.getSourceOp(slice);
            sliceOp->set<unsigned>("opId", initialOpId);
            slice->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);
            op.getOutputTensor()[0]->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);
            op.getOutputTensor()[0]->set<bool>("noPropagate", true);
            om.undefineFlow(inputFlow);
            nextOp->setInputTensor(slice, idx, false);
            om.defineFlow(slice, nextOp, idx);
        }
    }
    return;
}
