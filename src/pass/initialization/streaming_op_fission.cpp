#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/edge_exists.hpp"
#include <algorithm>
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"


static void streamingOpFissionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(StreamingOpFission)
        .setFunc(streamingOpFissionFcn)
        .setDescription(
            "This pass splits an operation into multiple smaller ops, to be run over time"
        );

    }

}

void streamingOpFissionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Info, "STREAMING PASS entering pass");

    // original graph
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    // get ops to split and number of splits from descriptor
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        pass.log(mv::Logger::MessageType::Info, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    // each s refers to an op
    for (auto s: strategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter") ;
        // TODO add heuristic to determine numsplits if no json
        // TODO is the ops only for Conv? If not then we need cases for each Op
        // or functions per Op type to allow copying passing its attribute on

        int32_t numWtSplits = s.get<int32_t>("weight_splits") ;
        auto opToFracture = om.getOp(nodeName);
        auto opType = opToFracture->getOpType();
        unsigned currentOpId = opToFracture->get<unsigned>("opId");
        pass.log(mv::Logger::MessageType::Info, "Streaming Pass: splitting Op " + nodeName + " into " + std::to_string(numWtSplits));

        auto OpAttrDF = opToFracture->get("dilationFactor");
        auto OpAttrGrp = opToFracture->get("group");
        auto OpAttrPad = opToFracture->get("padding");
        auto OpAttrStride = opToFracture->get("stride");
        auto OpAttrQPs = opToFracture->get("quantParams");

        auto parentTensor = opToFracture->getInputTensor(0);

        // gather original weights attributes
        auto originalWtsTensor = opToFracture->getInputTensor(1);
        auto originalWeights = originalWtsTensor->getData();
        auto WtsAttrDType = originalWtsTensor->get("dType");
        auto WtsAttrShape = originalWtsTensor->get("shape");
        auto WtsAttrOrder = originalWtsTensor->get("order");
        auto shapeArray = WtsAttrShape.get<mv::Shape>();
        int numKernels = shapeArray[3];

        // build replacement sub graph
        int kWtSize = originalWeights.size()/numKernels;
        std::vector<mv::Data::TensorIterator> opsToJoin(numWtSplits) ;     // resulting ops after fission

        size_t startIndex = 0;
        size_t endIndex = 0;
        size_t biasStartIndex = 0;
        size_t biasEndIndex = 0;
        size_t numKernelsPerSplit = mv::ceil_division(numKernels, numWtSplits);
        size_t remainingKernels = numKernels;
        for (size_t wi=0; wi < numWtSplits; wi++)
        {
            size_t subWtSize = originalWeights.size();

            //How many kernels in this sub-Op
            numKernelsPerSplit = std::min(numKernelsPerSplit, remainingKernels);
            remainingKernels -= numKernelsPerSplit;

            shapeArray[3] = numKernelsPerSplit;
            subWtSize = kWtSize * numKernelsPerSplit;

            startIndex = endIndex;
            endIndex = startIndex + subWtSize ;
            if ( endIndex > originalWeights.size())
            {
                endIndex = originalWeights.size();
            }

            pass.log(mv::Logger::MessageType::Info, "Streaming Pass: wtssize beg,end = " + std::to_string(subWtSize) + " " + std::to_string(startIndex) + " , " +  std::to_string(endIndex));

            std::vector<mv::DataElement>::const_iterator first = originalWeights.begin() + startIndex;
            std::vector<mv::DataElement>::const_iterator last = originalWeights.begin() + endIndex;
            std::vector<mv::DataElement> subWt(first, last);

            std::string newName = nodeName+std::to_string(wi);

            // TODO generalize tensor shape, order (assumes wts N=shape(3) )
            mv::Data::TensorIterator weightsX ;

            if (originalWtsTensor->hasAttr("quantParams"))
            {
                auto WtsAttrQPs = originalWtsTensor->get("quantParams");
                weightsX = om.constantDataElement(subWt, shapeArray, WtsAttrDType, WtsAttrOrder, WtsAttrQPs, newName+"weights#1");
            }
            else
            {
                weightsX = om.constantDataElement(subWt, shapeArray,  WtsAttrDType, WtsAttrOrder, {{}, {}, {}, {}}, newName+"weights#1");
            }
            auto originalOpId = om.getSourceOp(originalWtsTensor)->get<unsigned>("opId");
            om.getSourceOp(weightsX)->set<unsigned>("opId", originalOpId);
            auto convX = om.conv(parentTensor, weightsX, OpAttrStride, OpAttrPad, OpAttrDF, OpAttrGrp, OpAttrQPs, newName);

            if (opToFracture->hasAttr("bias"))
            {
                biasStartIndex = biasEndIndex;
                biasEndIndex = biasStartIndex + numKernelsPerSplit;

                auto biasTensorName = opToFracture->get<std::string>("bias");
                auto originalBiasTensor = dm.getTensor(biasTensorName);
                auto oiginalBiasData = originalBiasTensor->getData();
                if ( biasEndIndex > oiginalBiasData.size())
                {
                    biasEndIndex = oiginalBiasData.size();
                }
                std::vector<mv::DataElement>::const_iterator biasFirst = oiginalBiasData.begin() + biasStartIndex;
                std::vector<mv::DataElement>::const_iterator biasLast = oiginalBiasData.begin() + biasEndIndex;
                std::vector<mv::DataElement> subBiasData(biasFirst, biasLast);

                std::string newBiasTensorName = mv::createBiasName(newName);
                mv::Data::TensorIterator biasTensor;

                mv::Data::TensorIterator biasTensorX;
                if (originalBiasTensor->hasAttr("quantParams"))
                {
                    auto biasAttrQPs = originalBiasTensor->get("quantParams");
                    biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {numKernelsPerSplit}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData, biasAttrQPs ));
                }
                else
                {
                    biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {numKernelsPerSplit}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData));
                }

                om.addAttr(om.getSourceOp(convX), "bias", biasTensorX->getName());
                pass.log(mv::Logger::MessageType::Info, "Copied Bias attribute of Fractured conv " + opToFracture->getName() + " to " + convX->getName());
            }

            om.getSourceOp(convX)->set<unsigned>("opId", currentOpId);

            opsToJoin[wi] = convX ;
        }
        mv::Data::TensorIterator subGraphOut;
        if (opToFracture->hasAttr("quantParams"))
        {
            subGraphOut = om.concat(opsToJoin,"C", opToFracture->get("quantParams"));
        }
        else
        {
            subGraphOut = om.concat(opsToJoin,"C");
        }
        //TODO consider moving the pass till after "ComputeTensorQuantParams" then concat will get the quantization params for output tensor
        om.getSourceOp(subGraphOut)->set<unsigned>("opId", currentOpId);
        // Create lists of children ops, and the input slot used into each
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        for (mv::Data::FlowSiblingIterator sinkFlow(opToFracture.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
        }

        // remove original fractured op and the constant tensors input to it
        om.removeOp(om.getSourceOp(originalWtsTensor));
        om.removeOp(opToFracture);

        // reconnect subgraph output to children
        for (unsigned j = 0; j < opsToLink.size(); ++j)
        {
            opsToLink[j]->setInputTensor(subGraphOut, inputSlots[j]);
            om.defineFlow(subGraphOut, opsToLink[j], inputSlots[j]);
            //std::cout << "IN STREAMING PASS: new input " <<  inputSlots[j] << " of " << opsToLink[j]->getName() << " is " << opsToLink[j]->getInputTensor(0)->getName() << std::endl ;
        }

    }
    pass.log(mv::Logger::MessageType::Info, "STREAMING PASS done");
}