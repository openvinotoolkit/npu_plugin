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

    std::cout << "STREAMING PASS entering pass" << std::endl;
    // original graph
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // get ops to split and number of splits from descriptor
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        std::cout << "STREAMING PASS EXITING: no strategy defined in JSON" << std::endl;
        pass.log(mv::Logger::MessageType::Info, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    // each s refers to an op
    for (auto s: strategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter") ;
// TODO add heuristic to determine numsplits if no json
        int32_t numWtSplits = s.get<int32_t>("weight_splits") ;
        auto opToFracture = om.getOp(nodeName);
        std::cout << "IN STREAMING PASS: splitting op " << nodeName << " into " << numWtSplits << std::endl ;
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

        int startIndex = 0;
        int endIndex = 0;
        for (int wi=0; wi<numWtSplits; wi++)
        {
            int subWtSize = originalWeights.size();
            shapeArray[3]=numKernels/numWtSplits;
            // for odd splits, first modulo wts tensors will be larger by 1 K-size
            if (wi<numKernels % numWtSplits)
            {
                subWtSize = kWtSize * (numKernels/numWtSplits+1);
                shapeArray[3]=numKernels/numWtSplits+1; 
            }
            // even splits or rest of odd wts tensors after modulus
            else
            {
                subWtSize = kWtSize * (numKernels/numWtSplits);
                shapeArray[3]=numKernels/numWtSplits;
            }

            startIndex = endIndex;
            endIndex=startIndex+subWtSize ;
            if ( endIndex > originalWeights.size())
            {
                endIndex = originalWeights.size();
            } 
            std::cout << "In STREAMING PASS: wtssize beg,end = " << subWtSize << "   " << startIndex << "," << endIndex << std::endl ;
            std::vector<mv::DataElement>::const_iterator first = originalWeights.begin() + startIndex;
            std::vector<mv::DataElement>::const_iterator last = originalWeights.begin() + endIndex;
            std::vector<mv::DataElement> subWt(first, last);
            std::string newName = nodeName+"_"+std::to_string(wi);
            auto WtsAttrDType = originalWtsTensor->get("dType");
            auto WtsAttrShape = originalWtsTensor->get("shape");
// TODO generalize tensor shape, order (assumes wts N=shape(3) )
            auto WtsAttrOrder = originalWtsTensor->get("order");
            mv::Data::TensorIterator weightsX ;
            if (parentTensor->hasAttr("quantParams"))
            {
                auto WtsAttrQPs = parentTensor->get("quantParams");
                weightsX = om.constantDataElement(subWt, shapeArray, WtsAttrDType, WtsAttrOrder, WtsAttrQPs, newName+"wts");
            }
            else
            {
                weightsX = om.constantDataElement(subWt, shapeArray,  WtsAttrDType, WtsAttrOrder, {{}, {}, {}, {}}, newName+"wts");
            }
            auto convX = om.conv(parentTensor, weightsX, OpAttrStride, OpAttrPad, OpAttrDF, OpAttrGrp, OpAttrQPs, newName);

            if (opToFracture->hasAttr("bias"))
            {
                auto biasTensorName = opToFracture->get<std::string>("bias");
                om.addAttr(om.getSourceOp(convX), "bias", biasTensorName);
                pass.log(mv::Logger::MessageType::Info, "Copied Bias attribute of Fractured conv " + opToFracture->getName() + " to " + convX->getName());
            }
            
            opsToJoin[wi] = convX ;
        }
        auto subGraphOut = om.concat(opsToJoin);
//       auto subGraphOut = om.concat(opsToJoin,"C", {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}, {23}, {24581}});

        // Create lists of children ops, and the input slot used into each
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        for (mv::Data::FlowSiblingIterator sinkFlow(opToFracture.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
        }
  
        // remove original fractured op and the constant tensors input to it
        while(opToFracture.parentsSize() > 1)
        {
            auto paramOp = opToFracture.leftmostParent();
            ++paramOp;
            om.removeOp(paramOp);
        }
        om.removeOp(opToFracture);
    
        // reconnect subgraph output to children
        for (unsigned j = 0; j < opsToLink.size(); ++j)
        {
            opsToLink[j]->setInputTensor(subGraphOut, inputSlots[j]);
            om.defineFlow(subGraphOut, opsToLink[j], inputSlots[j]);
            std::cout << "IN STREAMING PASS: new input " <<  inputSlots[j] << " of " << opsToLink[j]->getName() << " is " << opsToLink[j]->getInputTensor(0)->getName() << std::endl ;
        }

    }
    std::cout << "EXIT STREAMING PASS" << std::endl ;
}