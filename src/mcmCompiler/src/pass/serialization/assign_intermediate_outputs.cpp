//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void assignIntermediateOutputsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AssignIntermediateOutputNodes)
        .setFunc(assignIntermediateOutputsFcn)
        .setDescription(
            "This pass assigns user defined nodes to be intermediate outputs to be dumped to disk by runtime for debug."
        );
    }
}

void assignIntermediateOutputsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    std::vector<std::string> intermediateNodes;
    if (passDesc.hasAttr("tensors"))
        intermediateNodes = passDesc.get<std::vector<std::string>>("tensors");
    else
        return; // exit pass

    // add new output nodes
    int numOutputIndex = om.getNumNetworkOutputs();
    std::vector<std::pair<mv::Data::TensorIterator, mv::Data::OpListIterator>> newOutputs;
    for (std::string node : intermediateNodes)
    {
        // find exact tensor
        mv::Data::TensorIterator tensor = om.findTensor(node);
        if (tensor == om.tensorEnd()) 
        {
            pass.log(mv::Logger::MessageType::Warning, "Tensor name not found! Please check name in Compilation Descriptor (are you missing ':0' at end?)");
            continue;
        }
        // assign an output operation to it
        mv::Data::TensorIterator intermediateOutput = om.output(tensor->getName() + "_output", tensor, tensor->get<mv::DType>("dType"), true);
        mv::Data::OpListIterator outputOp = om.getNetworkOutput(om.getNumNetworkOutputs() - 1);
        newOutputs.push_back(std::make_pair(tensor, outputOp));
    }

    // check if any new outputs to process
    if (newOutputs.size() == 0)
        return;

    // convert new outputs to ImplicitOutput
    std::vector<mv::Data::TensorIterator> newImplicitOutputTensors;
    for (auto newOutputIt = newOutputs.begin(); newOutputIt != newOutputs.end(); ++newOutputIt)
    {
        mv::Data::TensorIterator intermediateTensor = newOutputIt->first; 
        mv::Data::OpListIterator newOutputOp = newOutputIt->second;
        
        // create implicit output
        mv::Data::TensorIterator implicitOutput = om.implicitOutput("intermediate_out", intermediateTensor);
        implicitOutput->set<uint8_t>("outputIndex", numOutputIndex);
        implicitOutput->set<mv::DType>("precision", intermediateTensor->get<mv::DType>("dType"));
        om.getSourceOp(implicitOutput)->set<uint8_t>("outputIndex", numOutputIndex);
        om.getSourceOp(implicitOutput)->set<std::string>("networkOutputName", newOutputOp->getName());
        om.getSourceOp(implicitOutput)->set<mv::DType>("precision", intermediateTensor->get<mv::DType>("dType"));

        newImplicitOutputTensors.push_back(implicitOutput);

        // replace output node and remove all references
        auto inputFlow = newOutputOp.leftmostInput();
        om.replaceNetworkOutputAtIdx(numOutputIndex, om.getSourceOp(implicitOutput));
        om.undefineFlow(inputFlow);
        om.removeOp(newOutputOp);
        numOutputIndex++;
    }

    
    // check if implicitUnion already exists
    std::vector<mv::Data::OpListIterator> outputUnions = om.getOps("ImplicitUnion");
    if (outputUnions.size() > 0 )
    {
        mv::Data::OpListIterator outputUnion = outputUnions[0];
        size_t countInputs = outputUnion->getInputTensor().size();
        // add the new impclitOutput nodes as inputTensors to union and define flow
        for (auto tensorIt = newImplicitOutputTensors.begin(); tensorIt != newImplicitOutputTensors.end(); ++tensorIt)
        {
            outputUnion->addInputTensor(*tensorIt);
            om.defineFlow(*tensorIt, outputUnion, countInputs++);
        }
    }
    else 
    {   
        // Replace existing Output node with an ImplicitOutput
        mv::Data::OpListIterator networkOutput = om.getNetworkOutput(0);
        mv::Data::TensorIterator inputTensor = networkOutput->getInputTensor(0);
        mv::Data::TensorIterator implicitOutput = om.implicitOutput("final_output", inputTensor);
        
        implicitOutput->set<uint8_t>("outputIndex", 0);
        // implicitOutput->set<std::set<std::string>>("allocators", {"ProgrammableOutput", } );
        implicitOutput->set<mv::DType>("precision", inputTensor->get<mv::DType>("dType"));
        om.getSourceOp(implicitOutput)->set<uint8_t>("outputIndex", 0);
        om.getSourceOp(implicitOutput)->set<std::string>("networkOutputName", inputTensor->getName());
        om.getSourceOp(implicitOutput)->set<mv::DType>("precision", inputTensor->get<mv::DType>("dType"));

        newImplicitOutputTensors.insert(newImplicitOutputTensors.begin(), implicitOutput);

        // remove the flow going into original output node        
        auto inputFlow = networkOutput.leftmostInput();
        om.replaceNetworkOutputAtIdx(0, om.getSourceOp(implicitOutput));

        // remove all references to original output node
        om.undefineFlow(inputFlow);
        om.removeOp(networkOutput);
        
        // create ImplicitUnion and connect to a newly created output node
        mv::Data::TensorIterator outputUnion = om.implicitUnion("impl_union", newImplicitOutputTensors);
        auto output = om.output("out_union", outputUnion, mv::DType("Default"), false);
    }
}
