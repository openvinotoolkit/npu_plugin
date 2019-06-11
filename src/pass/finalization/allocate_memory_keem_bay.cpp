#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

static void allocateGraphfileTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void allocateCMXTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
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

        MV_REGISTER_PASS(AllocateGraphfileTensorsKeemBay)
        .setFunc(allocateGraphfileTensorsFcnKeemBay)
        .setDescription(
            "Perform allocation of all populated tensors using memory allocator"
        );

        MV_REGISTER_PASS(AllocateCMXTensorsKeemBay)
        .setFunc(allocateCMXTensorsFcnKeemBay)
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
void allocateGraphfileTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
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
        if (opType == "Constant" || opType == "ConstantInt" || opType == "ConstantDataElement" || opType == "WeightsTable" || opType == "SparsityMap")
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
void allocateCMXTensorsFcnKeemBay(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
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
        else if (opType == "Concat")
        {
            // Allocate Output
            auto outputTensor = opIterator->getOutputTensor(0);
            if (outputTensor->hasAttr("allocators"))
                dm.deallocateTensor("VPU_CMX_NN", stageIt, outputTensor);

            auto outputBuffer = dm.allocateTensor("VPU_CMX_NN", stageIt, outputTensor);

            // Allocate Inputs inside of that output
            unsigned valid_inputs = opIterator->inputSlots();
            auto concat_axis_index = mv::Shape::getAxis(opIterator->get<std::string>("axis"));
            std::vector<unsigned> running_concat_offset_LHS;
            auto prev_offset = 0;
            auto offset = 0;
            for(unsigned i = 0; i != valid_inputs; i++){
                running_concat_offset_LHS.push_back(prev_offset + offset);
                prev_offset = prev_offset + offset;
                // Calculate for next tensor
                offset = opIterator->getInputTensor(i)->getShape()[concat_axis_index];
            }

            std::vector<unsigned> running_concat_offset_RHS;
            std::copy(running_concat_offset_LHS.begin(),
                    running_concat_offset_LHS.end(),
                    back_inserter(running_concat_offset_RHS));
            std::reverse(std::begin(running_concat_offset_RHS), std::end(running_concat_offset_RHS));

            //std::cout << "running_concat_offset_LHS: ";
            //for(auto i : running_concat_offset_LHS)
            //    std::cout << i<< ",";
            //std::cout << std::endl;
            //std::cout << "running_concat_offset_RHS: ";
            //for(auto i : running_concat_offset_RHS)
            //    std::cout << i<< ",";
            //std::cout << std::endl;
            //std::cout << "Output Tensor Shape: " << outputTensor->getShape().toString() << std::endl;

            for(unsigned i = 0; i != valid_inputs; i++){
                auto inputTensor = opIterator->getInputTensor(i);

                // If already allocated from a previous pass, deallocate.
                // Note: This is probably not a good long term solution as we may have
                // requirements from two different connections, this approach only resolves one.
                // Probably restrictions on a tensor should be attributes of that tensor.
                if (!inputTensor->hasAttr("allocators"))
                    dm.allocateTensor("VPU_CMX_NN", stageIt, inputTensor);

                std::vector<std::size_t> lhs_padding(inputTensor->getShape().ndims());
                std::vector<std::size_t> rhs_padding(inputTensor->getShape().ndims());


                // This code assumes all tensors are of equal size. TODO: Assertions
                auto lhs = running_concat_offset_LHS[i];
                auto rhs = running_concat_offset_RHS[i];

                lhs_padding.at(concat_axis_index) = lhs;
                rhs_padding.at(concat_axis_index) = rhs;

                auto ExistingBuffer = dm.getBuffer("VPU_CMX_NN", stageIt, inputTensor);


                //std::cout << "Tensor Shape: " << inputTensor->getShape().toString() << std::endl;
                //std::cout << "\t\tLeft Padding: ";
                //for(auto i : lhs_padding)
                //    std::cout << i<< ",";
                //std::cout << std::endl;
                //std::cout << "\t\tRight Padding: ";
                //for(auto i : rhs_padding)
                //    std::cout << i<< ",";
                //std::cout << std::endl;

                auto NewBuffer = dm.moveTensor("VPU_CMX_NN", ExistingBuffer, outputBuffer, lhs_padding, rhs_padding);

            }
        }
        /*
            For each input and output, allocate if it has not already been done.
            Don't allocate for I/O layers as they are already accounted for.
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
