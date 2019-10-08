#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"

void checkTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element& compOutput);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(CheckTensors)
        .setFunc(checkTensorsFcn)
        .setLabel("Debug")
        .setDescription(
            "Check if tensors stored in the computation model are in a valid state."
            "States that are considered to be invalid:"
            " - Populated tensor is an output of a non const operation"
            " - Unpopulated tensor is an output of a const operation"
            " - Populated tensor is referenced by a data flow that source is not a const op"
            " - Populated tensor holding no data"
        );

    }

}

void checkTensorsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element& compOutput)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    compOutput.set<bool>("valid", true);
    compOutput.set<std::vector<std::string>>("unpopulatedConstOpOutput", {});
    compOutput.set<std::vector<std::string>>("populatedNonConstOpOutput", {});
    compOutput.set<std::vector<std::string>>("populatedNonConstDataFlow", {});
    compOutput.set<std::vector<std::string>>("emptyPopulated", {});

    OpModel om(model);

    for (auto it = om.opBegin(); it != om.opEnd(); ++it)
    {

        if (it->getOpType() == "Constant")
        {

            

            if (!it->getOutputTensor(0)->isPopulated())
            {

                compOutput.set<bool>("valid", false);
                compOutput.get<std::vector<std::string>>("unpopulatedConstOpOutput").push_back(it->getOutputTensor(0)->getName());

            }
            
        }
        else
        {

            for (std::size_t i = 0; i < it->outputSlots(); ++i)
            {

                if (it->getOutputTensor(i)->isPopulated())
                {

                    compOutput.set<bool>("valid", false);
                    compOutput.get<std::vector<std::string>>("populatedNonConstOpOutput").push_back(it->getOutputTensor(0)->getName());

                }

            }
            
        }

    }

    DataModel dm(model);

    for (auto it = dm.flowBegin(); it != dm.flowEnd(); ++it)
    {

        if (it->getTensor()->isPopulated() && it.source()->getOpType() != "Constant")
        {

            compOutput.set<bool>("valid", false);
            compOutput.get<std::vector<std::string>>("populatedNonConstDataFlow").push_back(it->getTensor()->getName());
        
        }

    }

    for (auto it = dm.tensorBegin(); it != dm.tensorEnd(); ++it)
    {

        if (it->isPopulated() && it->getShape().totalSize() == 0)
        {

            compOutput.set<bool>("valid", false);
            compOutput.get<std::vector<std::string>>("emptyPopulated").push_back(it->getName());
            
        }

    }

}