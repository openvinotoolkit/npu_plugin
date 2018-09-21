#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"

static void allocatePopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
// static void allocateForImplicitConcat();


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

// void allocateForImplicitConcat(){
//
// }

void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("IntermediateMemory"))
        throw ArgumentError(dm, "allocator", "IntermediateMemory", "Computation model does not have IntermediateMemory specified");

    if (cm.stageSize() == 0)
        throw ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");


    // We iterate through the tensors because some may not be directly attached to an Op
    // were we to use that iterator instead.
    for (auto tIt = dm.tensorBegin(); tIt != dm.tensorEnd(); ++tIt)
    {

        OpModel om(dm) ;
        bool external = false, fake = false;
        std::vector<std::string> input_names, output_names, invalid_names, c_pad_names;

        int max_pad = 0;

        for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
        {

            if (opIterator->getOpType() == OpType::Concat){
                auto in0 = opIterator->getInputTensor(0)->getName();
                auto in1 = opIterator->getInputTensor(1)->getName();
                auto out = opIterator->getOutputTensor(0)->getName();

                // If already allocated, must be deallocated so that we can stride properly.
                // Note: This is probably not a good long term solution as we may have
                // requirements from two different connections, this approach only resolves one.
                // Probably restrictions on a tensor should be attributes of that tensor.
                if (in0.hasAttr("allocated") && in0.get<bool>("allocated") == true){
                    dm.deallocateTensor("IntermediateMemory", stageIt, in0);
                }
                if (in1.hasAttr("allocated") && in1.get<bool>("allocated") == true){
                    dm.deallocateTensor("IntermediateMemory", stageIt, in1);
                }
                if (out.hasAttr("allocated") && out.get<bool>("allocated") == true){
                    dm.deallocateTensor("IntermediateMemory", stageIt, out);
                }

                // auto outputBuf = m.allocate(outputTensor, 0);
                // auto input1Buf = m.allocate(inputTensor1, outputBuf, {0, 0, 0}, {0, 0, 2});
                // auto input2Buf = m.allocate(inputTensor2, outputBuf, {0, 0, 2}, {0, 0, 0});

                // std::cout << outputBuf->second->toString(true) << std::endl;
                // std::cout << input1Buf->second->toString(true) << std::endl;
                // std::cout << input2Buf->second->toString(true) << std::endl;

                dm.allocateTensor("IntermediateMemory", stageIt, tIt, paddings);
                dm.allocateTensor("IntermediateMemory", stageIt, tIt, paddings);
                dm.allocateTensor("IntermediateMemory", stageIt, tIt, paddings);
            }


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

        if (std::find(invalid_names.begin(), invalid_names.end(), tIt->getName()) != invalid_names.end())
        {
            std::cout << "Condition Invalid Hit" << std::endl;
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
