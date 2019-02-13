#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

static void allocatePopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void allocateUnpopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void allocateInputOutputTensors(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

// static void allocateForImplicitConcat();


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AllocateInputOutputTensors)
        .setFunc(allocateInputOutputTensors)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "Perform allocation of all input and output tensors using memory allocator"
        );

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

//NOTE:This pass assumes the existence of memory allocators called ProgrammableInput and ProgrammableOutput. Lucklily this is true for both MX and Keembay.
void allocateInputOutputTensors(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    std::cout << "Allocate input/output tensors" << std::endl;

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("ProgrammableInput")){
        throw ArgumentError(dm, "allocator", "ProgrammableInput", "Computation model does not have ProgrammableInput specified");
    }

    if (!dm.hasAllocator("ProgrammableOutput")){
        throw ArgumentError(dm, "allocator", "ProgrammableOutput", "Computation model does not have ProgrammableOutput specified");
    }

    if (cm.stageSize() == 0){
        throw ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");
    }


    OpModel om(dm) ;
    //bool external = false;
    std::vector<std::string> external_names;
    auto stageIt = cm.getStage(0);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        for (unsigned x = 0; x < opIterator->inputSlots(); x++)
        {
            auto inTensor = opIterator->getInputTensor(x);
            if (!inTensor->isPopulated() &&
                (! inTensor->hasAttr("allocator")) &&
                (inTensor->hasAttr("modelInput") && inTensor->get<bool>("modelInput")))
            {
                auto buf = dm.allocateTensor("ProgrammableInput", stageIt, inTensor);
                if (opIterator->hasAttr("NCE1_Compatible"))
                {
                    if (opIterator->get<int>("NCE1_Compatible"))
                    {
                        if (inTensor->hasAttr("NCE1_Paddings"))
                        {
                            std::cout << "Padding for hardware" << std::endl;
                            auto paddings = inTensor->get<std::vector<std::size_t>>("NCE1_Paddings");
                            dm.padRight("ProgrammableInput", buf, paddings);

                        }
                    }
                }

            }
        }
        for (unsigned x = 0; x < opIterator->outputSlots(); x++)
        {

            auto outTensor = opIterator->getOutputTensor(x);
            if (!outTensor->isPopulated() &&
                (! outTensor->hasAttr("allocator")) &&
                (outTensor->hasAttr("modelOutput") && outTensor->get<bool>("modelOutput"))
                )
            {
                auto buf = dm.allocateTensor("ProgrammableOutput", stageIt, outTensor);
                if (opIterator->hasAttr("NCE1_Compatible"))
                {
                    if (opIterator->get<int>("NCE1_Compatible"))
                    {
                        if(outTensor->hasAttr("NCE1_Paddings"))
                        {
                            std::cout << "Padding for hardware" << std::endl;
                            auto paddings = outTensor->get<std::vector<std::size_t>>("NCE1_Paddings");
                            dm.padRight("ProgrammableOutput", buf, paddings);

                        }
                    }
                }

            }

        }



    }

    std::cout << "Exiting allocate input and output" << std::endl;
}

// NOTE:This pass assumes the existence of a memory allocator called ConstantMemory. This was true only for MX, so this pass has to be changed to use TargetDescriptor.
// By doing so, we will know what Memories the Target provides for Populated Tensors
// Also, NCE1_Paddings shall become NCE_Paddings.
void allocatePopulatedTensorsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    std::cout << "Allocate populated" << std::endl;

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("ConstantMemory"))
        throw ArgumentError(dm, "allocator", "ConstantMemory", "Computation model does not have ConstantMemory specified");

    if (cm.stageSize() == 0)
        throw ArgumentError(cm, "stages count", "0", "Computation model does not have stages specified");

    auto stageIt = cm.getStage(0);

    for (auto tIt = dm.tensorBegin(); tIt != dm.tensorEnd(); ++tIt)
    {

        if (tIt->isPopulated())
        {


            if(tIt->hasAttr("NCE1_Paddings"))
            {

                std::vector<std::size_t> rhs_paddings = tIt->get<std::vector<std::size_t>>("NCE1_Paddings");


                mv::Shape original_shape = tIt->getShape();
                std::vector<std::size_t> original_shape_v = original_shape;
                std::vector<std::size_t> padded_shape_v = std::vector<std::size_t>(original_shape.ndims());
                for(std::size_t i = 0; i< original_shape_v.size(); i++)
                    padded_shape_v[i] += original_shape_v[i] + rhs_paddings[i];

                auto buf = dm.allocateTensor("ConstantMemory", stageIt, tIt);
                dm.padRight("ConstantMemory",  buf, rhs_paddings);

            }else{
                auto buf = dm.allocateTensor("ConstantMemory", stageIt, tIt);
            }

        }

    }

    std::cout << "Exiting allocate populated" << std::endl;

}


// NOTE:This pass assumes the existence of a memory allocator called IntermediateMemory. This was true only for MX, so this pass has to be changed to use TargetDescriptor to see
// which allocators are available for IntermediateMemory.
void allocateUnpopulatedTensorsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    std::cout << "Allocate unpopulated" << std::endl;

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("IntermediateMemory")){
        throw ArgumentError(dm, "allocator", "IntermediateMemory", "Computation model does not have IntermediateMemory specified");
    }

    if (cm.stageSize() == 0){
        throw ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");
    }


    OpModel om(dm) ;
    //bool external = false;
    std::vector<std::string> external_names;
    auto stageIt = cm.getStage(0);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (opIterator->getOpType() == "Concat")
        {
            // Allocate Output
            auto outputTensor = opIterator->getOutputTensor(0);
            if (outputTensor->hasAttr("allocator"))
                dm.deallocateTensor("IntermediateMemory", stageIt, outputTensor);

            auto outputBuffer = dm.allocateTensor("IntermediateMemory", stageIt, outputTensor);

            // Allocate Inputs inside of that output
            unsigned valid_inputs = opIterator->inputSlots();

            //auto axis = opIterator->get<int>("axis");
            unsigned int channel_index = 0;

            // TODO: Request this from the order - What element is axis at.
            // Currently only working for channels.
            // auto inOrder = opIterator->getInputTensor(0)->getOrder();
            channel_index = 2;

            std::vector<unsigned> running_concat_offset_LHS;
            auto prev_offset = 0;
            auto offset = 0;
            for(unsigned i = 0; i != valid_inputs; i++){
                running_concat_offset_LHS.push_back(prev_offset + offset);
                prev_offset = prev_offset + offset;
                // Calculate for next tensor
                offset = opIterator->getInputTensor(i)->getShape()[channel_index];
            }

            std::vector<unsigned> running_concat_offset_RHS;
            std::copy(running_concat_offset_LHS.begin(),
                    running_concat_offset_LHS.end(),
                    back_inserter(running_concat_offset_RHS));
            std::reverse(std::begin(running_concat_offset_RHS), std::end(running_concat_offset_RHS));

            // std::cout << "running_concat_offset_LHS: ";
            // for(auto i : running_concat_offset_LHS)
            //     std::cout << i<< ",";
            // std::cout << std::endl;
            // std::cout << "running_concat_offset_RHS: ";
            // for(auto i : running_concat_offset_RHS)
            //     std::cout << i<< ",";
            // std::cout << std::endl;

            // std::cout << "Output Tensor Shape: " << outputTensor->getShape().toString() << std::endl;

            for(unsigned i = 0; i != valid_inputs; i++){
                auto inputTensor = opIterator->getInputTensor(i);

                // If already allocated from a previous pass, deallocate.
                // Note: This is probably not a good long term solution as we may have
                // requirements from two different connections, this approach only resolves one.
                // Probably restrictions on a tensor should be attributes of that tensor.
                if (!inputTensor->hasAttr("allocator"))
                    dm.allocateTensor("IntermediateMemory", stageIt, inputTensor);

                std::vector<std::size_t> lhs_padding(inputTensor->getShape().ndims());
                std::vector<std::size_t> rhs_padding(inputTensor->getShape().ndims());


                // This code assumes all tensors are of equal size. TODO: Assertions
                auto lhs = running_concat_offset_LHS[i];
                auto rhs = running_concat_offset_RHS[i];

                lhs_padding.at(channel_index) = lhs;
                rhs_padding.at(channel_index) = rhs;

                auto ExistingBuffer = dm.getBuffer("IntermediateMemory", stageIt, inputTensor);


                // std::cout << "Tensor Shape: " << inputTensor->getShape().toString() << std::endl;
                // std::cout << "\t\tLeft Padding: ";
                // for(auto i : lhs_padding)
                //     std::cout << i<< ",";
                // std::cout << std::endl;
                // std::cout << "\t\tRight Padding: ";
                // for(auto i : rhs_padding)
                //     std::cout << i<< ",";
                // std::cout << std::endl;

                auto NewBuffer = dm.moveTensor("IntermediateMemory", ExistingBuffer, outputBuffer, lhs_padding, rhs_padding);

            }
        }
        else if (opIterator->getOpType() == "Input")
        {
            std::cout << "Input" << std::endl;
            auto outTensor = opIterator->getOutputTensor(0);
            outTensor->set<bool>("modelInput", true);
            if(outTensor->hasAttr("allocator"))
            {
                std::cout << "Deallocate Input" << std::endl;
                dm.deallocateTensor("IntermediateMemory", stageIt, outTensor);
            }

        }
        else if (opIterator->getOpType() == "Output")
        {
            std::cout << "Output" << std::endl;
            auto inTensor = opIterator->getInputTensor(0);
            inTensor->set<bool>("modelOutput", true);
            if(inTensor->hasAttr("allocator"))
            {
                std::cout << "Deallocate Output" << std::endl;
                dm.deallocateTensor("IntermediateMemory", stageIt, inTensor);
            }

        }
        /*
            For each input and output, allocate if it has not already been done.
            Don't allocate for Concat or I/O layers as they are already accounted for.
        */
        else
        {
            for (unsigned x = 0; x < opIterator->inputSlots(); x++)
            {

                auto inTensor = opIterator->getInputTensor(x);

                if (!inTensor->isPopulated() &&
                    (! inTensor->hasAttr("allocator")) &&
                    (! inTensor->hasAttr("modelInput") || ! inTensor->get<bool>("modelInput")) &&
                    (! inTensor->hasAttr("modelOutput") || ! inTensor->get<bool>("modelOutput"))
                    )
                {

                    auto buf = dm.allocateTensor("IntermediateMemory", stageIt, inTensor);
                    if (opIterator->hasAttr("NCE1_Compatible"))
                    {

                        if (opIterator->get<int>("NCE1_Compatible"))
                        {

                            if (inTensor->hasAttr("NCE1_Paddings"))
                            {

                                std::cout << "Padding for hardware" << std::endl;
                                auto paddings = inTensor->get<std::vector<std::size_t>>("NCE1_Paddings");
                                dm.padRight("IntermediateMemory", buf, paddings);

                            }

                        }

                    }

                }
            }
            for (unsigned x = 0; x < opIterator->outputSlots(); ++x)
            {

                auto outTensor = opIterator->getOutputTensor(x);
                if (!outTensor->isPopulated() &&
                    (! outTensor->hasAttr("allocator")) &&
                    (! outTensor->hasAttr("modelInput") || ! outTensor->get<bool>("modelInput")) &&
                    (! outTensor->hasAttr("modelOutput") || ! outTensor->get<bool>("modelOutput"))
                    )
                {

                    auto buf = dm.allocateTensor("IntermediateMemory", stageIt, outTensor);
                    if (opIterator->hasAttr("NCE1_Compatible"))
                    {

                        if (opIterator->get<int>("NCE1_Compatible"))
                        {

                            if(outTensor->hasAttr("NCE1_Paddings"))
                            {

                                std::cout << "Padding for hardware" << std::endl;
                                auto paddings = outTensor->get<std::vector<std::size_t>>("NCE1_Paddings");
                                dm.padRight("IntermediateMemory", buf, paddings);

                            }

                        }

                    }

                }

            }

        }

    }

    std::cout << "Exiting allocate unpopulated" << std::endl;

}
