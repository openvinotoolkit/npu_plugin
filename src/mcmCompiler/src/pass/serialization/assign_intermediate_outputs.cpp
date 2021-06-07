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
#include <regex>

static constexpr size_t MAX_TENSOR_SIZE_ALLOWED = 8388608; // 8MB
static constexpr int MAX_DDR = 2097152; // 2MB

static void assignIntermediateOutputsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {
        /* Usage: pass should be added after G.O. decisions have been saved (e.g. before or after streaming passes)
                  `op_names` specifies the exact names of the operations whose output tensor will be dumped
                  `op_filter` allows filtering the operations using regular expressions
                  At least one of the options should be used.
        {
            "name" : "AssignIntermediateOutputNodes",
            "op_names" : ["fire2/expand1x1_1", "fire2/expand3x3_1", "pool3"],
            "op_filters" : ["pool.*"],
            "max_ddr_use" : 2097152,
            "_comment" : "Runtime will dump these intermediate tensors to disk for debug. Specify max size of DDR to consume, e.g. 2MB"
        },
        */
        MV_REGISTER_PASS(AssignIntermediateOutputNodes)
        .setFunc(assignIntermediateOutputsFcn)
        .setDescription(
            "This pass assigns user defined nodes to be intermediate outputs to be dumped to disk by runtime for debug."
        );
    }
}

bool isTensorValid(const mv::pass::PassEntry& pass, mv::OpModel& om, const mv::Data::TensorIterator& tensor, int& ddrUsed, const int ddrMax)
{
    // Parent op produces runtime sparse tensor
    auto sourceOp = om.getSourceOp(tensor);
    if (sourceOp->hasAttr("outputActivationSparsity") && sourceOp->get<bool>("outputActivationSparsity")) {
        pass.log(mv::Logger::MessageType::Debug, "Cannot dump sparse tensor: " + tensor->getName());
        return false;
    }

    // Check if tensor is already an output
    bool isAlreadyOutput = false;
    for (auto childOp = sourceOp.leftmostChild(); childOp != om.opEnd(); ++childOp) {
        if (childOp->getOpType() == "Output" || childOp->getOpType() == "ImplicitOutput") {
            isAlreadyOutput = true;
            break;
        }
    }
    if (isAlreadyOutput) {
        pass.log(mv::Logger::MessageType::Debug, "Op is already an output: " + tensor->getName());
        return false;
    }

    // Output ops are retained in DDR till end of network, so limit amount the of DDR used
    const size_t tensorSize = tensor->getShape().totalSize() * tensor->getDType().getSizeInBytes();
    if (tensorSize > MAX_TENSOR_SIZE_ALLOWED) {
        pass.log(mv::Logger::MessageType::Debug, "Cannot attach output to Op: " + tensor->getName() + " due to size: " +
                    std::to_string(tensorSize) + " bytes. Max tensor size allowed: " + std::to_string(MAX_TENSOR_SIZE_ALLOWED));
        return false;
    }
    ddrUsed += tensorSize;
    if (ddrUsed > ddrMax) {
        pass.log(mv::Logger::MessageType::Debug, "Maximum DDR usage has been reached. No more ops will be dumped");
        return false;
    }

    return true;
}

mv::Data::OpListIterator createImplicitOutput(mv::OpModel& om, const mv::Data::TensorIterator& tensor, const uint8_t outputIndex, const mv::DType& dType)
{
    const auto parentOp = om.getSourceOp(tensor);
    auto implicitOutput = om.implicitOutput(parentOp->getName() + "_output", tensor);
    auto implicitOutputOp = om.getSourceOp(implicitOutput);
    implicitOutput->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
    implicitOutput->set<uint8_t>("outputIndex", outputIndex);
    implicitOutput->set<mv::DType>("precision", dType);
    implicitOutputOp->set<uint8_t>("outputIndex", outputIndex);
    implicitOutputOp->set<mv::DType>("precision", dType);
    implicitOutputOp->set<std::string>("networkOutputName", parentOp->getName());
    implicitOutputOp->set<bool>("propagateLocation", false);
    implicitOutputOp->set<unsigned>("opId", parentOp->get<unsigned>("opId"));
    return implicitOutputOp;
}

void linkNewNetworkOutputs(mv::OpModel& om, std::vector<mv::Data::TensorIterator>& newImplicitOutputTensors, std::set<uint8_t>& previousOutputIndices)
{
    auto implicitUnionOps = om.getOps("ImplicitUnion");
    if (!implicitUnionOps.empty())
    {
        // Model already has multiple outputs; link new outputs to ImplicitUnion
        auto implicitUnion = implicitUnionOps[0];

        // Store the indices of the previous outputs
        for (auto implicitOutput = implicitUnion.leftmostParent(); implicitOutput != om.opEnd(); ++implicitOutput)
            previousOutputIndices.insert(implicitOutput->get<uint8_t>("outputIndex"));

        // Add the new ImplicitOutput nodes as input tensors to union and define flows
        size_t countInputs = implicitUnion->inputSlots();
        for (auto& newImplicitOutput : newImplicitOutputTensors)
        {
            implicitUnion->addInputTensor(newImplicitOutput);
            om.defineFlow(newImplicitOutput, implicitUnion, countInputs++);
        }
    }
    else
    {
        // Model has only one output; create ImplicitUnion and link all outputs
        previousOutputIndices.insert(0);

        // Replace existing Output node with an ImplicitOutput
        auto networkOutputOp = om.getNetworkOutput(0);
        auto inputTensor = networkOutputOp->getInputTensor(mv::IO_TENSOR_INPUT);
        const auto implicitOutputOp = createImplicitOutput(om, inputTensor, 0, networkOutputOp->get<mv::DType>("precision"));
        newImplicitOutputTensors.insert(newImplicitOutputTensors.begin(), implicitOutputOp->getOutputTensor(0));

        // Remove the original output node
        auto inputFlow = networkOutputOp.leftmostInput();
        om.replaceNetworkOutputAtIdx(0, implicitOutputOp);
        om.undefineFlow(inputFlow);
        om.removeOp(networkOutputOp);

        // Create ImplicitUnion and connect all new ImplicitOutput nodes
        auto outputUnion = om.implicitUnion("impl_union", newImplicitOutputTensors);
        om.output("output_union", outputUnion, mv::DType("Default"), false);
    }
}

void assignIntermediateOutputsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    if (!passDesc.hasAttr("op_names") && !passDesc.hasAttr("op_filters"))
        return;

    int ddrUsed = 0;
    const int ddrMax = passDesc.hasAttr("max_ddr_use") ? passDesc.get<int>("max_ddr_use") : MAX_DDR;

    // Extract the names of operations from the pass attributes
    std::vector<std::string> intermediateNodes;
    if (passDesc.hasAttr("op_names"))
        intermediateNodes = passDesc.get<std::vector<std::string>>("op_names");
    if (passDesc.hasAttr("op_filters"))
    {
        const auto filters = passDesc.get<std::vector<std::string>>("op_filters");
        const auto ops = om.topologicalSort();
        for (const auto& filter : filters)
        {
            for (const auto& op : ops)
            {
                const auto opName = op->getName();
                if (std::find(intermediateNodes.cbegin(), intermediateNodes.cend(), opName) == intermediateNodes.cend() &&
                    std::regex_match(opName, std::regex(filter)))
                    intermediateNodes.push_back(opName);
            }
        }
    }
    if (intermediateNodes.empty())
    {
        pass.log(mv::Logger::MessageType::Debug, "No operations found");
        return;
    }

    // Find the tensors produced by the requested operations
    std::vector<mv::Data::TensorIterator> newOutputTensors;
    for (const auto& node : intermediateNodes)
    {
        auto tensor = om.findTensor(node + ":0");
        if (tensor == om.tensorEnd())
        {
            // Maybe it was changed into a streaming op and is now a concat (of streams)
            tensor = om.findTensor(node + "concat_:0");
            if (tensor == om.tensorEnd()) {
                pass.log(mv::Logger::MessageType::Debug, "Tensor for op '" + node + "' not found! Op not processed for output");
                continue;
            }
        }

        if (isTensorValid(pass, om, tensor, ddrUsed, ddrMax))
            newOutputTensors.push_back(tensor);
    }
    if (newOutputTensors.empty())
    {
        pass.log(mv::Logger::MessageType::Debug, "No tensors to set as output");
        return;
    }

    // Add the new intermediate outputs
    std::vector<mv::Data::TensorIterator> newImplicitOutputTensors;
    for (const auto& newOutput : newOutputTensors)
    {
        pass.log(mv::Logger::MessageType::Debug, "Adding intermediate output: " + newOutput->getName());
        const auto outputIndex = om.getNumNetworkOutputs();
        const auto implicitOutputOp = createImplicitOutput(om, newOutput, outputIndex, newOutput->getDType());
        om.addNetworkOutput(implicitOutputOp);
        newImplicitOutputTensors.push_back(implicitOutputOp->getOutputTensor(0));
    }

    // Link the new intermediate outputs to the network ImplicitUnion
    std::set<uint8_t> previousOutputIndices;
    linkNewNetworkOutputs(om, newImplicitOutputTensors, previousOutputIndices);

    // Print new indices for output ops
    pass.log(mv::Logger::MessageType::Debug, "New output indices:");
    for (const auto& outputOp : om.getNetworkOutputs())
    {
        const auto outputIndex = outputOp->get<uint8_t>("outputIndex");
        const auto tensorSize = outputOp->getOutputTensor(0)->getShape().totalSize() * outputOp->getOutputTensor(0)->getDType().getSizeInBytes();
        if (previousOutputIndices.find(outputIndex) != previousOutputIndices.end())
            pass.log(mv::Logger::MessageType::Debug, "  " + std::to_string(outputIndex) + ": " + outputOp->getName() +
                " (" + std::to_string(tensorSize) + " bytes, existing output)");
        else
            pass.log(mv::Logger::MessageType::Debug, "  " + std::to_string(outputIndex) + ": " + outputOp->getName() +
                " (" + std::to_string(tensorSize) + " bytes, new output)");
    }
}
