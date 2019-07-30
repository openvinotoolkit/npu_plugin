#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"

static void fuseReluMXFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(FuseReluMX)
        .setFunc(fuseReluMXFcn)
        .setDescription(
            "Fuses a relu op to the parent op."
            "Relu op is removed from the model, and a new attribute of type OpType and value relu is defined for parent Op."
        );

    }

}

void fuseReluMXFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    using namespace mv;
    OpModel om(model);

    std::cout << "Fusing ReLU to HW operations" << std::endl;

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Relu")
        {

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            if(!parentOpIt->hasAttr("NCE1_Compatible"))
                continue;
            if(!parentOpIt->get<int>("NCE1_Compatible"))
                continue;
            parentOpIt->set<std::string>("postOpType", "Relu");

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                std::size_t inputIdx = sinkFlow->get<std::size_t>("sinkInput");
                sinkFlow.sink()->erase("input" + std::to_string(inputIdx));
                om.defineFlow(sourceTensor, sinkFlow.sink(), inputIdx);
            }

            while(opIt.parentsSize() > 1)
            {
                auto paramOp = opIt.leftmostParent();
                ++paramOp;
                om.removeOp(paramOp);
            }

            om.removeOp(opIt);
            opIt = parentOpIt;

        }

    }

    std::cout << "Finished to fuse ReLU to hw operations" << std::endl;


}
