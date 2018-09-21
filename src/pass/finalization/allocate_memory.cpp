#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"

static void allocatePopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AllocatePopulatedTensors)
        .setFunc(allocatePopulatedTensorsFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "Perform allocation of all populated tensors using memory allocator"
        );

        MV_REGISTER_PASS(AllocateUnpopulatedTensors)
        .setFunc(allocateUnpopulatedTensorsFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "Perform allocation of all unpopulated tensors using memory allocator"
        );

    }

}

void allocatePopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("ConstantMemory"))
        throw ArgumentError(dm, "allocator", "ConstantMemory", "Computation model does not have ConstantMemory specified");

    if (cm.stageSize() == 0)
        throw ArgumentError(cm, "stages count", "0", "Computation model does not have stages specified");

    for (auto tIt = dm.tensorBegin(); tIt != dm.tensorEnd(); ++tIt)
    {
        if (tIt->isPopulated())
        {
            auto stageIt = cm.getStage(0);
            dm.allocateTensor("ConstantMemory", stageIt, tIt);

        }
    }

}

void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("IntermediateMemory"))
        throw ArgumentError(dm, "allocator", "IntermediateMemory", "Computation model does not have IntermediateMemory specified");

    if (cm.stageSize() == 0)
        throw ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");

    for (auto tIt = dm.tensorBegin(); tIt != dm.tensorEnd(); ++tIt)
    {

        OpModel om(dm) ;
        bool external = false, fake = false, conv_padding = false;
        std::vector<std::string> input_names, output_names, invalid_names, c_pad_names;

        int max_pad = 0;

        for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
        {
            if (opIterator->getOpType() == OpType::Input)
            {
                auto b = opIterator->getOutputTensor(0)->getName();
                input_names.push_back(b);
            }
            else if(opIterator->getOpType() == OpType::Output)
            {
                auto b = opIterator->getInputTensor(0)->getName();
                output_names.push_back(b);
            }
            else if(opIterator->getOpType() == OpType::Constant)
            {
                auto b = opIterator->getOutputTensor(0)->getName();
                invalid_names.push_back(b);
            }
            else if(opIterator->getOpType() == OpType::Conv2D)
            {
                auto ot_name = opIterator->getOutputTensor(0)->getName();
                if(tIt->getName() == ot_name)
                {

                    c_pad_names.push_back(ot_name);

                    int halfksizerounded = ((((opIterator->getInputTensor(1)->getShape()[0])/2)+1)*2);
                    int cpad = halfksizerounded*opIterator->getOutputTensor(0)->getShape()[1]*opIterator->getOutputTensor(0)->getShape()[2]*2;

                    if (cpad > max_pad)
                    {
                        max_pad = cpad;
                        max_pad = 0; // TODO: Actual Allocation
                    }

                }

            }

        }

        if (std::find(input_names.begin(), input_names.end(), tIt->getName()) != input_names.end())
        {
            external = true;
        }
        else
        {
            if(std::find(output_names.begin(), output_names.end(), tIt->getName()) != output_names.end())
            {
                external = true;
            }
            else
            {
                // Not external, dont do anything
            }
        }

        if (std::find(c_pad_names.begin(), c_pad_names.end(), tIt->getName()) != c_pad_names.end())
        {
            conv_padding = true;
        }
        else
        {
            // Not conv_padding, dont do anything
        }

        if (std::find(invalid_names.begin(), invalid_names.end(), tIt->getName()) != invalid_names.end())
        {
            fake = true;
        }


        if (!tIt->isPopulated() and !external and !fake)
        {
            
            auto stageIt = cm.getStage(0);

            int pad = 0;
            if (conv_padding){
                pad = max_pad;
            }

            dm.allocateTensor("IntermediateMemory", stageIt, tIt);

        }

    }

}
