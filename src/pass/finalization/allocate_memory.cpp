#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"

static void allocatePopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void allocateInputOutputTensors(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

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




void allocateInputOutputTensors(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
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
        std::cout << opIterator->getOpType().toString() << std::endl;
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

void allocatePopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
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
                for(int i =0; i< original_shape_v.size();i++)
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



void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
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

        if (opIterator->getOpType() == OpType::Concat)
        {

            auto in0 = opIterator->getInputTensor(0);
            auto in1 = opIterator->getInputTensor(1);
            auto out = opIterator->getOutputTensor(0);

            // If already allocated, must be deallocated so that we can stride properly.
            // Note: This is probably not a good long term solution as we may have
            // requirements from two different connections, this approach only resolves one.
            // Probably restrictions on a tensor should be attributes of that tensor.

            if (!in0->hasAttr("allocator")){
                dm.allocateTensor("IntermediateMemory", stageIt, in0);
            }
            if (!in1->hasAttr("allocator")){
                dm.allocateTensor("IntermediateMemory", stageIt, in1);
            }
            if (out->hasAttr("allocator")){
                dm.deallocateTensor("IntermediateMemory", stageIt, out);
            }

            auto outRef = dm.allocateTensor("IntermediateMemory", stageIt, out);

            //TODO: assert equal amount of dimensions and equal layouts.
            std::vector<std::size_t> empty_padding(in0->getShape().ndims());
            std::vector<std::size_t> lhs_padding(in0->getShape().ndims());
            std::vector<std::size_t> rhs_padding(in0->getShape().ndims());


            auto axis = opIterator->get<int>("axis");
            unsigned int channel_index = 0;

            // TODO: I think there is a gap in functionality here that would make this trivial

            switch(in0->getOrder())
            {
                case OrderType::RowMajor:
                {
                    switch(axis){
                        case 2: // Channels
                        {
                            channel_index = 2;
                        }
                        break;
                        default:
                        {
                            std::cout << "Concat not supported for this axis" << std::endl;
                            assert(0);
                        }
                    }
                }
                break;
                case OrderType::RowMajorPlanar:
                {
                    switch(axis){
                        case 2: // Channels
                        {
                            channel_index = 2;
                        }
                        break;
                        default:
                        {
                            std::cout << "Concat not supported for this axis" << std::endl;
                            assert(0);
                        }
                    }
                }
                break;
                case OrderType::ColumnMajorPlanar:
                {
                    switch(axis){
                        case 2: // Channels
                        {
                            channel_index = 0;
                        }
                        break;
                        default:
                        {
                            std::cout << "Concat not supported for this axis" << std::endl;
                            assert(0);
                        }
                    }
                }
                break;
                case OrderType::RowInterleaved:
                {
                    switch(axis){
                        case 2: // Channels
                        {
                            channel_index = 2;
                        }
                        break;
                        default:
                        {
                            std::cout << "Concat not supported for this axis" << std::endl;
                            assert(0);
                        }
                    }
                }
                break;
                case OrderType::ColumnMajor:
                {
                    switch(axis){
                        case 2: // Channels
                        {
                            channel_index = 0;
                        }
                        break;
                        default:
                        {
                            std::cout << "Concat not supported for this axis" << std::endl;
                            assert(0);
                        }
                    }
                }
                break;
                default:
                {
                    std::cout << "Order: "<< in0->getOrder().toString() << std::endl;
                    std::cout << "Concat not supported for this format" << std::endl;
                    assert(0);
                }
            }

            auto lhs = in0->getShape()[channel_index];
            auto rhs = in1->getShape()[channel_index];
            lhs_padding.at(channel_index) = lhs;
            rhs_padding.at(channel_index) = rhs;


            auto in0buf = dm.getBuffer("IntermediateMemory", stageIt, in0);
            auto in1buf = dm.getBuffer("IntermediateMemory", stageIt, in1);


            in0buf = dm.moveTensor("IntermediateMemory", in0buf, outRef, empty_padding, rhs_padding);
            in1buf = dm.moveTensor("IntermediateMemory", in1buf, outRef, lhs_padding, empty_padding);
            in0->set<unsigned>("Offset", in0buf->getOffset());
            in1->set<unsigned>("Offset", in1buf->getOffset());
            in0->set<unsigned>("LeadPad", in0buf->getStrides()[0]);
            in1->set<unsigned>("LeadPad", in1buf->getStrides()[0]);



        }
        else if (opIterator->getOpType() == OpType::Input)
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
        else if (opIterator->getOpType() == OpType::Output)
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

        else if(opIterator->getOpType() == mv::OpType::ReLU)
        {
            auto inTensor = opIterator->getInputTensor(0);
            auto outTensor = opIterator->getOutputTensor(0);
            std::vector<std::size_t> empty_padding(outTensor->getShape().ndims());

            dm.deallocateTensor("IntermediateMemory", stageIt, inTensor);
            auto outBuf = dm.allocateTensor("IntermediateMemory", stageIt, outTensor);
            dm.allocateTensor("IntermediateMemory", outBuf, inTensor, empty_padding, empty_padding);
        }
        /*
            For each input and output, allocate if it has not already been done.
            Don't allocate for Concat or I/O layers as they are already accounted for.
        */
        else
        {
            std::cout << opIterator->getOpType().toString() << std::endl;

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
            for (unsigned x = 0; x < opIterator->outputSlots(); x++)
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
