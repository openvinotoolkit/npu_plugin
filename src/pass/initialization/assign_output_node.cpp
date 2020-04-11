#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void assignOutputFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AssignOutputNode)
        .setFunc(assignOutputFcn)
        .setDescription(
            "This pass assigns the output node of the computational data graph, to support multiple outputs."
        );
    }
}

void assignOutputFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    if (om.getNumNetworkOutputs() == 0)
    {
        std::cout << "error: zero network outputs specified" << std::endl;
        return;
    }

    if (om.getNumNetworkOutputs() == 1)
    {
        om.setOutputNode(om.getNetworkOutput(0));
        return;
    }

    auto networkOutputs = om.getNetworkOutputs();

    std::vector<mv::Data::TensorIterator> outputTensors;
    for (size_t i = 0; i < networkOutputs.size(); i++)
    {
        auto inputTensor = networkOutputs[i]->getInputTensor(0);
        // Assumes one input per outputNode
        auto implicitOutput = om.implicitOutput(inputTensor);
        om.getSourceOp(implicitOutput)->set<uint8_t>("outputIndex", i);
        outputTensors.push_back(implicitOutput);
        outputTensors[i]->set<uint8_t>("outputIndex", i);
        auto inputFlow = networkOutputs[i].leftmostInput();
        om.replaceNetworkOutputAtIdx(i, om.getSourceOp(implicitOutput));

        // remove all references to the original output nodes
        om.undefineFlow(inputFlow);
        om.removeOp(networkOutputs[i]);
    }


    // Create an implicit union, connect all network outputs to that union, and attach;

    auto outputUnion = om.implicitUnion(outputTensors);
    auto output = om.output(outputUnion, mv::DType("Default"), {{},{},{},{}}, false);
}
