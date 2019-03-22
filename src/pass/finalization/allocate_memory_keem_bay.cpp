#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

static void allocatePopulatedTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void allocateUnpopulatedTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void allocateInputOutputTensorsKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

// static void allocateForImplicitConcat();


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AllocateInputOutputTensorsKeemBay)
        .setFunc(allocateInputOutputTensorsKeemBay)
        .setDescription(
            "Perform allocation of all input and output tensors using memory allocator"
        );

        MV_REGISTER_PASS(AllocatePopulatedTensorsKeemBay)
        .setFunc(allocatePopulatedTensorsFcnKeemBay)
        .setDescription(
            "Perform allocation of all populated tensors using memory allocator"
        );

        MV_REGISTER_PASS(AllocateUnpopulatedTensorsKeemBay)
        .setFunc(allocateUnpopulatedTensorsFcnKeemBay
        )
        .setDescription(
            "Perform allocation of all unpopulated tensors using memory allocator"
        );
    }
}

/* Tensors from Graph input/output operations are stored in:
 * 1) ProgrammableInput
 * 2) ProgrammableOutput
*/
void allocateInputOutputTensorsKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Allocating input/output tensors");

    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    if (!dm.hasAllocator("ProgrammableInput"))
        throw mv::ArgumentError(dm, "allocator", "ProgrammableInput", "Computation model does not have ProgrammableInput specified");


    if (!dm.hasAllocator("ProgrammableOutput"))
        throw mv::ArgumentError(dm, "allocator", "ProgrammableOutput", "Computation model does not have ProgrammableOutput specified");


    if (cm.stageSize() == 0)
        throw mv::ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");

    mv::OpModel om(dm);
    auto stageIt = cm.getStage(0);

    auto inputOp = om.getInput();
    auto inTensor = inputOp->getOutputTensor(0);

    if (!inTensor->isPopulated() && (! inTensor->hasAttr("allocators")))
        dm.allocateTensor("ProgrammableInput", stageIt, inTensor);

    auto outputOp = om.getOutput();
    auto outTensor = outputOp->getInputTensor(0);

    if (!outTensor->isPopulated() && (! outTensor->hasAttr("allocators")))
        dm.allocateTensor("ProgrammableOutput", stageIt, outTensor);
}

//Populated Tensors are stored in:
// 1) GraphFile
//
void allocatePopulatedTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Allocating populated tensors");

    mv::ControlModel cm(model);
    mv::DataModel dm(model);
    mv::OpModel om(model);
    
    if (!dm.hasAllocator("GraphFile"))
         throw mv::ArgumentError(dm, "allocator", "GraphFile", "Computation model does not have GraphFile allocator specified");

    if (cm.stageSize() == 0)
         throw mv::ArgumentError(cm, "stages count", "0", "Computation model does not have stages specified");

    auto stageIt = cm.getStage(0);

    unsigned i = 0;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();
        if (opType == "Constant" || opType == "ConstantInt" || opType == "ConstantDataElement")
        {
            auto tIt = opIterator->getOutputTensor(0);
            dm.allocateTensor("GraphFile", stageIt, tIt);
            tIt->set<unsigned>("graphFileIndex", i++);
        }
    }
}

/* Unpopulated Tensors are stored in:
 * 1) VPU_CMX_NN
 * 2) VPU_DDR_BSS
*/
void allocateUnpopulatedTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    pass.log(mv::Logger::MessageType::Debug, "Allocating unpopulated tensors");

    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    if (!dm.hasAllocator("VPU_CMX_NN"))
        throw mv::ArgumentError(dm, "allocator", "VPU_CMX_NN", "Computation model does not have VPU_CMX_NN specified");

    if (!dm.hasAllocator("VPU_DDR_BSS"))
        throw mv::ArgumentError(dm, "allocator", "VPU_DDR_BSS", "Computation model does not have VPU_DDR_BSS specified");

    if (cm.stageSize() == 0)
        throw mv::ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");

    mv::OpModel om(dm);
    auto stageIt = cm.getStage(0);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();
        if (opType == "Input")
        {
            auto outTensor = opIterator->getOutputTensor(0);
            outTensor->set<bool>("modelInput", true); /*Assign tensor attribute  modelInput"*/
        }

        else if (opType == "Output")
        {
            auto inTensor = opIterator->getInputTensor(0);
            inTensor->set<bool>("modelOutput", true); /*Assign tensor attribute  modelOutput"*/
    
        }

        else if (opType == "Constant" || opType == "ConstantInt" || opType == "ConstantDataElement")
            continue;

        /*
            For each input and output, allocate if it has not already been done.
            Don't allocate for Concat or I/O layers as they are already accounted for.
        */
        else
        {
            for (unsigned x = 0; x < opIterator->inputSlots(); x++)
            {

                auto inTensor = opIterator->getInputTensor(x);

                if (
                    (! inTensor->hasAttr("allocators")) &&
                    (! inTensor->hasAttr("modelInput") || ! inTensor->get<bool>("modelInput")) &&
                    (! inTensor->hasAttr("modelOutput") || ! inTensor->get<bool>("modelOutput"))
                    )
                {
                    dm.allocateTensor("VPU_CMX_NN", stageIt, inTensor);
                }
            }
            for (unsigned x = 0; x < opIterator->outputSlots(); ++x)
            {

                auto outTensor = opIterator->getOutputTensor(x);
                std::cout << outTensor->getName() << std::endl << std::endl;;
                if (
                    (! outTensor->hasAttr("allocators")) &&
                    (! outTensor->hasAttr("modelInput") || ! outTensor->get<bool>("modelInput")) &&
                    (! outTensor->hasAttr("modelOutput") || ! outTensor->get<bool>("modelOutput"))
                    )
                {
                    dm.allocateTensor("VPU_CMX_NN", stageIt, outTensor);
                }
            }
        }
    }
 }
