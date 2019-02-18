#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

static void allocatePopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void allocateUnpopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void allocateInputOutputTensors(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

// static void allocateForImplicitConcat();


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AllocateInputOutputTensorsKeemBay)
        .setFunc(allocateInputOutputTensors)
        .setDescription(
            "Perform allocation of all input and output tensors using memory allocator"
        );

        MV_REGISTER_PASS(AllocatePopulatedTensorsKeemBay)
        .setFunc(allocatePopulatedTensorsFcn)
        .setDescription(
            "Perform allocation of all populated tensors using memory allocator"
        );

        MV_REGISTER_PASS(AllocateUnpopulatedTensorsKeemBay)
        .setFunc(allocateUnpopulatedTensorsFcn)
        .setDescription(
            "Perform allocation of all unpopulated tensors using memory allocator"
        );
    }
}

/* Tensors from Graph input/output operations are stored in:
 * 1) ProgrammableInput
 * 2) VPU_DDR_Heap
 * 3) ProgrammableOutput
*/
void allocateInputOutputTensorsKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Allocating input/output tensors");

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("ProgrammableInput")){
        throw ArgumentError(dm, "allocator", "ProgrammableInput", "Computation model does not have ProgrammableInput specified");
    }

    if (!dm.hasAllocator("ProgrammableOutput")){
        throw ArgumentError(dm, "allocator", "ProgrammableOutput", "Computation model does not have ProgrammableOutput specified");
    }

    if (!dm.hasAllocator("VPU_DDR_Heap")){
        throw ArgumentError(dm, "allocator", "VPU_DDR_Heap", "Computation model does not have VPU_DDR_Heap specified");
    }

    if (cm.stageSize() == 0){
        throw ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");
    }


    OpModel om(dm) ;
    auto stageIt = cm.getStage(0);

    auto inputOp = om.getInput();
    
    for (unsigned x = 0; x < inputOp->outputSlots(); x++)
    {
        auto inTensor = inputOp->getOutputTensor(x);

        if (!inTensor->isPopulated() && (! inTensor->hasAttr("allocators")))
        {
                auto buf0 = dm.allocateTensor("ProgrammableInput", stageIt, inTensor);
                auto buf1 = dm.allocateTensor("VPU_DDR_Heap", stageIt, inTensor);   
        }
    }

    auto outputOp = om.getOutput();

    for (unsigned x = 0; x < outputOp->inputSlots(); x++)
    {
        auto outTensor = outputOp->getInputTensor(x);

        if (!outTensor->isPopulated() && (! outTensor->hasAttr("allocators")))
        {
            auto buf0 = dm.allocateTensor("ProgrammableOutput", stageIt, outTensor);
            auto buf1 = dm.allocateTensor("VPU_DDR_Heap", stageIt, outTensor);
        }
    }
}

/* Populated Tensors are stored in:
 * 1) VPU_DDR_BSS
 * 2) GraphFile
*/
void allocatePopulatedTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Allocating populated tensors");

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("VPU_DDR_BSS"))
         throw ArgumentError(dm, "allocator", "CVPU_DDR_BSS", "Computation model does not have VPU_DDR_BSS specified");
    
    if (!dm.hasAllocator("GraphFile"))
         throw ArgumentError(dm, "allocator", "CVPU_DDR_BSS", "Computation model does not have VPU_DDR_BSS specified");

    if (cm.stageSize() == 0)
         throw ArgumentError(cm, "stages count", "0", "Computation model does not have stages specified");

    auto stageIt = cm.getStage(0);

    for (auto tIt = dm.tensorBegin(); tIt != dm.tensorEnd(); ++tIt)
    {
        if (tIt->isPopulated())
        {
            auto buf0 = dm.allocateTensor("VPU_DDR_BSS", stageIt, tIt);
            auto buf1 = dm.allocateTensor("GraphFile", stageIt, tIt);
        }

    }
}


/* Unpopulated Tensors are stored in:
 * 1) VPU_DDR_BSS
 * 2) GraphFile
*/
void allocateUnpopulatedTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    pass.log(mv::Logger::MessageType::Debug, "Allocating unpopulated tensors");

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("VPU_CMX_NN")){
        throw ArgumentError(dm, "allocator", "VPU_CMX_NN", "Computation model does not have VPU_CMX_NN specified");
    }

    if (cm.stageSize() == 0){
        throw ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");
    }


    OpModel om(dm) ;
    std::vector<std::string> external_names;
    auto stageIt = cm.getStage(0);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (opIterator->getOpType() == "Input")
        {
            auto outTensor = opIterator->getOutputTensor(0);
            outTensor->set<bool>("modelInput", true); /*Assign tensor attribute  modelInput"*/
        }

        else if (opIterator->getOpType() == "Output")
        {
            auto inTensor = opIterator->getInputTensor(0);
            inTensor->set<bool>("modelOutput", true); /*Assign tensor attribute  modelOutput"*/
    
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
                std::cout << inTensor->getName() << std::endl << std::endl;;

                if (!inTensor->isPopulated() &&
                    (! inTensor->hasAttr("allocators")) &&
                    (! inTensor->hasAttr("modelInput") || ! inTensor->get<bool>("modelInput")) &&
                    (! inTensor->hasAttr("modelOutput") || ! inTensor->get<bool>("modelOutput"))
                    )
                {
                    auto buf = dm.allocateTensor("VPU_CMX_NN", stageIt, inTensor);
                }
            }
            for (unsigned x = 0; x < opIterator->outputSlots(); ++x)
            {

                auto outTensor = opIterator->getOutputTensor(x);
                std::cout << outTensor->getName() << std::endl << std::endl;;
                if (!outTensor->isPopulated() &&
                    (! outTensor->hasAttr("allocators")) &&
                    (! outTensor->hasAttr("modelInput") || ! outTensor->get<bool>("modelInput")) &&
                    (! outTensor->hasAttr("modelOutput") || ! outTensor->get<bool>("modelOutput"))
                    )
                {
                    auto buf = dm.allocateTensor("VPU_CMX_NN", stageIt, outTensor);
                }
            }
        }
    }
 }
