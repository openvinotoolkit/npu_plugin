#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"

void checkTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(CheckTensors)
        .setFunc(checkTensorsFcn)
        .setGenre(PassGenre::Validation)
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

void checkTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object& compOutput)
{

    using namespace mv;

    compOutput["valid"] = true;
    compOutput["invalidTensors"] = json::Object();
    compOutput["invalidTensors"]["unpopulatedConstOpOutput"] = json::Array();
    compOutput["invalidTensors"]["populatedNonConstOpOutput"] = json::Array();
    compOutput["invalidTensors"]["populatedNonConstDataFlow"] = json::Array();
    compOutput["invalidTensors"]["emptyPopulated"] = json::Array();

    OpModel om(model);

    for (auto it = om.opBegin(); it != om.opEnd(); ++it)
    {

        if (it->getOpType() == OpType::Constant)
        {

            

            if (!it->getOutputTensor(0)->isPopulated())
            {

                compOutput["valid"] = false;
                compOutput["invalidTensors"]["unpopulatedConstOpOutput"].append(it->getOutputTensor(0)->isPopulated());

            }
            
        }
        else
        {

            for (std::size_t i = 0; i < it->outputSlots(); ++i)
            {

                if (it->getOutputTensor(i)->isPopulated())
                {

                    compOutput["valid"] = false;
                    compOutput["invalidTensors"]["populatedNonConstOpOutput"].append(it->getOutputTensor(0)->getName());

                }

            }
            
        }

    }

    DataModel dm(model);

    for (auto it = dm.flowBegin(); it != dm.flowEnd(); ++it)
    {

        if (it->getTensor()->isPopulated() && it.source()->getOpType() != OpType::Constant)
        {

            compOutput["valid"] = false;
            compOutput["invalidTensors"]["populatedNonConstDataFlow"].append(json::Object({{it->getTensor()->getName(), it->getName()}}));
        
        }

    }

    for (auto it = dm.tensorBegin(); it != dm.tensorEnd(); ++it)
    {

        if (it->isPopulated() && it->getShape().totalSize() == 0)
        {

            compOutput["valid"] = false;
            compOutput["invalidTensors"]["emptyPopulated"].append(it->getName());
        
        }

    }

}