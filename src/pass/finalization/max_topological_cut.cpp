#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void maxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MaxTopologicalCut)
        .setFunc(maxTopologicalCut)
        .setDescription(
            "Calculate max topological cut."
        );
    }
}

void maxTopologicalCut(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        std::cout << "Name: " << opIt->getName() << std::endl;
        //std::cout << " Output tensor size " << opIt->getOutputTensor()[0]->getShape().toString() << std::endl; 

    }


  
}
