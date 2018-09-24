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
            auto buf = dm.allocateTensor("ConstantMemory", stageIt, tIt);
            if(tIt->hasAttr("NCE1_Paddings"))
            {
                
                auto paddings = tIt->get<std::vector<std::size_t>>("NCE1_Paddings");
                dm.padRight("ConstantMemory", buf, paddings);

            }
            
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

            if (in0->hasAttr("allocator")){
                dm.deallocateTensor("IntermediateMemory", stageIt, in0);
            }
            if (in1->hasAttr("allocator")){
                dm.deallocateTensor("IntermediateMemory", stageIt, in1);
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

            // TODO: I think there is a gap in funcitonality here that would make this trivial

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

            // std::cout << "Shape: "<< in0->getShape().toString() << std::endl;

            auto b = dm.allocateTensor("IntermediateMemory", outRef, in0, lhs_padding, empty_padding);
            auto a = dm.allocateTensor("IntermediateMemory", outRef, in1, empty_padding, rhs_padding);

            // std::cout << "Testing out: " << outRef->toString() << std::endl;
            // std::cout << "Testing in1 : " << b->toString() << std::endl;
            // std::cout << "Testing in2 : " << a->toString() << std::endl;
        }

        if (opIterator->getOpType() == OpType::Input){
            auto outTensor = opIterator->getOutputTensor(0);
            if(outTensor->hasAttr("allocator")){
                std::cout << "Deallocate Input" << std::endl;
                dm.deallocateTensor("IntermediateMemory", stageIt, outTensor);
            }
            outTensor->set<bool>("external", true);
        }
        if (opIterator->getOpType() == OpType::Output){
            auto inTensor = opIterator->getInputTensor(0);
            std::cout << inTensor->toString() << std::endl;
            if(inTensor->hasAttr("allocator")){
                std::cout << "Deallocate Output" << std::endl;
                dm.deallocateTensor("IntermediateMemory", stageIt, inTensor);
            }
            inTensor->set<bool>("external", true);
        }

        /*
            For each input and output, allocate if it has not already been done.
            Don't allocate for Concat or I/O layers as they are already accounted for.
        */
        if (opIterator->getOpType() != OpType::Input &&         // Alternative Storage
            opIterator->getOpType() != OpType::Output &&        // Alternative Storage
            opIterator->getOpType() != OpType::Concat)          // Already Accounted for.
        {
            for ( unsigned x =0; x != opIterator->inputSlots(); x++)
            {
                auto inTensor = opIterator->getInputTensor(x);
                if (!inTensor->isPopulated() &&
                    (! inTensor->hasAttr("allocator")) &&
                    (! inTensor->hasAttr("external") || ! inTensor->get<bool>("external"))
                    )
                {
                    auto buf = dm.allocateTensor("IntermediateMemory", stageIt, inTensor);
                    if(inTensor->hasAttr("NCE1_Paddings"))
                    {
                        
                        auto paddings = inTensor->get<std::vector<std::size_t>>("NCE1_Paddings");
                        dm.padRight("IntermediateMemory", buf, paddings);

                    }

                    std::cout << "allocate A: " << buf->getOffset() << std::endl;
                }
            }
            for ( unsigned x = 0; x != opIterator->outputSlots(); x++)
            {
                auto outTensor = opIterator->getOutputTensor(x);
                if (!outTensor->isPopulated() &&
                    (! outTensor->hasAttr("allocator")) &&
                    (! outTensor->hasAttr("external") || outTensor->get<bool>("external") == false)
                    )
                {
                    auto buf = dm.allocateTensor("IntermediateMemory", stageIt, outTensor);
                    std::cout << "[" << outTensor->getName() << "]" << std::endl;
                    std::cout << "allocate B: " << buf->getOffset() << std::endl;
                    if(outTensor->hasAttr("NCE1_Paddings"))
                    {
                        
                        auto paddings = outTensor->get<std::vector<std::size_t>>("NCE1_Paddings");
                        dm.padRight("IntermediateMemory", buf, paddings);

                    }

                    std::cout << "[" << outTensor->getName() << "]" << std::endl;
                    std::cout << "allocate B: " << buf->getOffset() << std::endl;

                }

            }
            
        }

    }

}
