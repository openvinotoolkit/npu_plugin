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
        /* example from squeezenet network:
        {
            "name" : "AssignIntermediateOutputNodes",
            "op_names" : ["fire2/expand1x1_1", "fire2/expand3x3_1", "pool3"],
            "max_ddr_use" : 2097152,
            "_comment" : "Runtime will dump these intermediate ops to disk for debug. Specify max size of DDR to consume, eg 2mb"
        },
        */
        MV_REGISTER_PASS(AssignIntermediateOutputNodes)
        .setFunc(assignIntermediateOutputsFcn)
        .setDescription(
            "This pass assigns user defined nodes to be intermediate outputs to be dumped to disk by runtime for debug."
        );
    }
}

void assignIntermediateOutputsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    // no individual tensors greater than this value are currently allowed
    const size_t MAX_TENSOR_SIZE_ALLOWED = 8388608; // 8mb

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    std::vector<std::string> intermediateNodes;
    if (passDesc.hasAttr("op_names"))
        intermediateNodes = passDesc.get<std::vector<std::string>>("op_names");
    else
        return; // exit pass

    // output ops are retained in DDR till end of network, so limit amount of DDR used
    int32_t ddrUsed = 0;
    int32_t ddrMax = 2097152; // default 2mb
    if (passDesc.hasAttr("max_ddr_use"))
        ddrMax = passDesc.get<int32_t>("max_ddr_use");

    // add new output nodes
    int numOutputIndex = om.getNumNetworkOutputs();
    int outputIndex = numOutputIndex;
    std::vector<std::pair<mv::Data::TensorIterator, mv::Data::OpListIterator>> newOutputs;
    std::string warnings;
    for (const std::string& node : intermediateNodes)
    {
        // find exact tensor
        mv::Data::TensorIterator tensor = om.findTensor(node + ":0");
        if (tensor == om.tensorEnd()) 
        {
            // maybe it was changed into a streaming op and is now a concat (of streams)
            tensor = om.findTensor(node + "concat_:0");
            if (tensor == om.tensorEnd()) {
                warnings += "Op name '" + node + "' not found! Op not processed for output\r\n";
                continue;
            }
        }
        // concat, eltwise - must be added together in DDR so all feeding tensor's memory location set to DDR
        mv::Data::OpListIterator sourceOp = om.getSourceOp(tensor);
        if ((sourceOp->getOpType() == "Concat") || (sourceOp->getOpType() == "Eltwise") ) {
            sourceOp->set<bool>("avoid_cmx_concat", true);
            for (auto& t : sourceOp->getInputTensor())
                t->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::DDR);
        }
        // Unsupported ops: ops feeding a concat, Permute. These ops cannot read memory location "OUTPUT"
        auto nextOp = sourceOp.leftmostChild();
        if ((nextOp->getOpType() == "Concat") || (nextOp->getOpType() == "Permute")) {
            warnings += "Cannot attach output to Ops feeding a \"" + nextOp->getOpType() + "\". Cannot dump: " + node + "\r\n";
            continue;
        }
        // Unsupported ops: if op already feeds an Output or ImplicitOutput
        if ((nextOp->getOpType() == "Output") || (nextOp->getOpType() == "ImplicitOutput")) {
            warnings += "Op is already an output: " + node + "\r\n";
            continue;
        }
        // limit individual tensor sizes to MAX_TENSOR_SIZE_ALLOWED
        size_t tensorSize = tensor->getShape().totalSize() * tensor->getDType().getSizeInBytes();
        if (tensorSize > MAX_TENSOR_SIZE_ALLOWED) {
            warnings += "Cannot attach output to Op: " + node + " due to size:  " + std::to_string(tensorSize) + " bytes. Max tensor size allowed: " + std::to_string(MAX_TENSOR_SIZE_ALLOWED) + "\r\n";
            break;
        }
        // limit amount of DDR used to ddrMax var
        ddrUsed += tensorSize;
        if (ddrUsed > ddrMax) {
            warnings += "max_ddr_use value has been reached. No more ops will be dumped";
            break;
        }
        // assign an output operation to it
        std::cout << "    output[" << outputIndex++ << "]: " << node << " (" << tensorSize << " bytes)" << std::endl;
        mv::Data::TensorIterator intermediateOutput = om.output(tensor->getName() + "_", tensor, tensor->get<mv::DType>("dType"), true);
        mv::Data::OpListIterator outputOp = om.getNetworkOutput(om.getNumNetworkOutputs() - 1);
        newOutputs.push_back(std::make_pair(tensor, outputOp));
    }
    std::cout << warnings << std::endl;

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
        mv::Data::TensorIterator implicitOutput = om.implicitOutput(newOutputOp->getName() + "output", intermediateTensor);
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
        size_t countInputs = outputUnion->inputSlots();
        // add the new implicitOutput nodes as inputTensors to union and define flow
        for (auto tensorIt = newImplicitOutputTensors.begin(); tensorIt != newImplicitOutputTensors.end(); ++tensorIt)
        {
            outputUnion->addInputTensor(*tensorIt);
            om.defineFlow(*tensorIt, outputUnion, countInputs++);
        }
    }
    else 
    {   
        // Replace existing Output node with an ImplicitOutput
        mv::Data::OpListIterator networkOutput = om.getNetworkOutput(mv::IO_TENSOR_INPUT);
        mv::Data::TensorIterator inputTensor = networkOutput->getInputTensor(mv::IO_TENSOR_INPUT);
        mv::Data::TensorIterator implicitOutput = om.implicitOutput("final_output", inputTensor);
        
        //transfer attributes
        implicitOutput->set<uint8_t>("outputIndex", 0);
        // implicitOutput->set<std::set<std::string>>("allocators", {"ProgrammableOutput", } );
        implicitOutput->set<mv::DType>("precision", networkOutput->get<mv::DType>("precision"));
        om.getSourceOp(implicitOutput)->set<uint8_t>("outputIndex", 0);
        om.getSourceOp(implicitOutput)->set<std::string>("networkOutputName", inputTensor->getName());
        om.getSourceOp(implicitOutput)->set<mv::DType>("precision", networkOutput->get<mv::DType>("precision"));
        om.getSourceOp(implicitOutput)->set<unsigned>("opId", networkOutput->get<unsigned>("opId"));

        newImplicitOutputTensors.insert(newImplicitOutputTensors.begin(), implicitOutput);

        // remove the flow going into original output node        
        auto inputFlow = networkOutput.leftmostInput();
        om.replaceNetworkOutputAtIdx(0, om.getSourceOp(implicitOutput));

        // remove all references to original output node
        om.undefineFlow(inputFlow);
        om.removeOp(networkOutput);
        
        // create ImplicitUnion and connect all new implicit output nodes
        mv::Data::TensorIterator outputUnion = om.implicitUnion("impl_union", newImplicitOutputTensors);
        auto output = om.output("output_union", outputUnion, mv::DType("Default"), false);
    }
}
