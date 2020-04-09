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
        outputTensors.push_back(om.implicitOutput(inputTensor));
        outputTensors[i]->set<uint8_t>("outputIndex", i);
        auto inputFlow = networkOutputs[i].leftmostInput();
        om.undefineFlow(inputFlow);
        om.removeOp(networkOutputs[i]);
    }

    // Flatten outputs if needed
    // Create an implicit concat, connect all network outputs to that concat, and attach;

    auto concat = om.concat(outputTensors, "N");
    auto output = om.output(concat, false);

    // TODO: use this API to assign graph output node. As of now, this is done when
    // generating the output node above.
    // om.setOutputNode(om.getSourceOp(output));
}
