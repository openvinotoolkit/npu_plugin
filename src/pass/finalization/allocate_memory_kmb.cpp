#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/flow/implicit_flow.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

static void allocateGraphfileTensorsKmbFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passArg, mv::Element&);
static void allocateCMXTensorsKmbFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void allocateInputOutputTensorsKmbFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void allocateImplicitOperationsKmbFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AllocateInputOutputTensorsKmb)
        .setFunc(allocateInputOutputTensorsKmbFcn)
        .setDescription(
            "Perform allocation of all input and output tensors using memory allocator"
        );


        MV_REGISTER_PASS(AllocateGraphfileTensorsKmb)
        .setFunc(allocateGraphfileTensorsKmbFcn)
        .setDescription(
            "Perform allocation of all populated tensors using memory allocator"
        );

        MV_REGISTER_PASS(AllocateCMXTensorsKmb)
        .setFunc(allocateCMXTensorsKmbFcn)
        .setDescription(
            "Perform allocation of all unpopulated tensors using memory allocator"
        );

        MV_REGISTER_PASS(ReAllocateImplicitOperationsKmb)
        .setFunc(allocateImplicitOperationsKmbFcn)
        .setDescription("Iterates over all implicit operations and moves implicit buffers into explicit buffers");
    }
}

/* Tensors from Graph input/output operations are stored in:
 * 1) ProgrammableInput
 * 2) ProgrammableOutput
*/
void allocateInputOutputTensorsKmbFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Allocating input/output tensors");

    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    if (!dm.hasAllocator("ProgrammableInput"))
        throw mv::ArgumentError(dm, "allocators", "ProgrammableInput", "Computation model does not have ProgrammableInput specified");


    if (!dm.hasAllocator("ProgrammableOutput"))
        throw mv::ArgumentError(dm, "allocators", "ProgrammableOutput", "Computation model does not have ProgrammableOutput specified");


    if (cm.stageSize() == 0)
        throw mv::ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");

    mv::OpModel om(dm);
    auto stageIt = cm.getStage(0);

    for(auto tensorIterator = om.tensorBegin(); tensorIterator != om.tensorEnd(); ++tensorIterator)
    {
        auto location = tensorIterator->get<mv::Tensor::MemoryLocation>("Location");
        if(location == mv::Tensor::MemoryLocation::INPUT)
        {
            dm.allocateTensor("ProgrammableInput", stageIt, tensorIterator);
        }
        else if(location == mv::Tensor::MemoryLocation::OUTPUT)
        {
            dm.allocateTensor("ProgrammableOutput", stageIt, tensorIterator);
        }
    }
}

//Populated Tensors are stored in:
// 1) GraphFile
void allocateGraphfileTensorsKmbFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Allocating populated tensors");

    mv::ControlModel cm(model);
    mv::DataModel dm(model);
    mv::OpModel om(model);

    unsigned numClusters = dm.getGlobalConfigParams()->get<int>("Number_of_Clusters");

    if (!dm.hasAllocator("GraphFile"))
         throw mv::ArgumentError(dm, "allocators", "GraphFile", "Computation model does not have GraphFile allocator specified");

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

            // Main tensor is always allocated to GraphFile
            // Subtensors are not
            dm.allocateTensor("GraphFile", stageIt, tIt);

            // For weights sparsity there is a seperate constant per cluster, the sub-tensors are sparsified individually to get the kernel data offsets 
            if(tIt->isSparse())
            {
                auto sparsityMap = tIt->getSparsityMap();
                auto sparsityMapIterator = dm.getTensor(sparsityMap->getName());
                dm.allocateTensor("GraphFile", stageIt, sparsityMapIterator);
                sparsityMap->set<unsigned>("graphFileIndex", i++);

                if(tIt->get<std::string>("splitStrategy") == "SplitOverK")
                {
                    for(std::size_t j = 0; j < numClusters; ++j)
                        tIt->getSubTensor(j).set<unsigned>("graphFileIndex", i++);
                }
                else
                    tIt->set<unsigned>("graphFileIndex", i++);
            }
            
            // SOK non-sparse weights are also serialised individually so that they can be compressed by the HDE 
            else if(tIt->get<std::string>("splitStrategy") == "SplitOverK" && !tIt->hasAttr("weightTable") && !tIt->hasAttr("sparsityMap"))   
            {
                for(std::size_t j = 0; j < numClusters; ++j)
                    tIt->getSubTensor(j).set<unsigned>("graphFileIndex", i++);
            }
            else
                tIt->set<unsigned>("graphFileIndex", i++);
        }
    }
}

// NOTE: This pass name is misleading. As a matter of fact, it allocates both populated and unpopulated tensors.
static mv::Data::BufferIterator allocateUnpopulatedTensor(const mv::pass::PassEntry& pass,mv::DataModel& dm,mv::Control::StageIterator& stageIt,mv::Data::TensorIterator& tensorIt)
{
    //todo:: stop with the if-else-if-else
    auto logicalLocation = tensorIt->get<mv::Tensor::MemoryLocation>("Location");
    if( logicalLocation == mv::Tensor::MemoryLocation::NNCMX)
    {
        auto toReturn = dm.allocateTensor("VPU_CMX_NN", stageIt, tensorIt);

        // Weights sparsity new approach
        if(tensorIt->isPopulated() && tensorIt->isSparse())
        {
            auto sparsityMap = tensorIt->getSparsityMap();
            auto sparsityMapIterator = dm.getTensor(sparsityMap->getName());
            dm.allocateTensor("VPU_CMX_NN", stageIt, sparsityMapIterator);
        }
        return toReturn;
    }
    else if(logicalLocation == mv::Tensor::MemoryLocation::UPACMX)
    {
        return dm.allocateTensor("VPU_CMX_UPA",stageIt, tensorIt);
    }
    else if(logicalLocation == mv::Tensor::MemoryLocation::DDR)
    {
        return dm.allocateTensor("VPU_DDR_Heap",stageIt, tensorIt);
    }
    else if(logicalLocation == mv::Tensor::MemoryLocation::BLOB)
    {
        return dm.allocateTensor("GraphFile",stageIt, tensorIt);
    }
    else if (logicalLocation == mv::Tensor::MemoryLocation::DEFAULT)
    {
        auto globalParams = dm.getGlobalConfigParams();
        std::string memoryLocation;
        if (globalParams->hasAttr("default_tensor_placement"))
        {
            mv::Tensor::MemoryLocation defaultPlace = mv::Tensor::MemoryLocation(globalParams->get<std::string>("default_tensor_placement"));
            if( defaultPlace == mv::Tensor::MemoryLocation::NNCMX)
            {
                memoryLocation = "VPU_CMX_NN";
            }
            else if(defaultPlace == mv::Tensor::MemoryLocation::UPACMX)
            {
                memoryLocation = "VPU_CMX_UPA";
            }
            else if(defaultPlace == mv::Tensor::MemoryLocation::DDR)
            {
                memoryLocation = "VPU_DDR_Heap";
            }
            pass.log(mv::Logger::MessageType::Debug, "Tensor " + tensorIt->getName() + " in default location. Allocating to " + memoryLocation + " as specified in json");
        }
        else
        {
            memoryLocation = "VPU_DDR_Heap";
            pass.log(mv::Logger::MessageType::Debug, "Tensor " + tensorIt->getName() + " in default location. Allocating to DDR_BSS as safety");
        }
        return dm.allocateTensor(memoryLocation, stageIt, tensorIt);
    }
    else
    {
        throw mv::AttributeError("Location:" , ": attempting to allocate unpopulated tensor to location " + logicalLocation.toString());
    }
}

/* Unpopulated Tensors are stored in:
 * 1) VPU_CMX_NN
 * 2) VPU_DDR_BSS
*/
void allocateCMXTensorsKmbFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Allocating unpopulated tensors");

    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    if (!dm.hasAllocator("VPU_CMX_NN"))
        throw mv::ArgumentError(dm, "allocators", "VPU_CMX_NN", "Computation model does not have VPU_CMX_NN specified");

    if (!dm.hasAllocator("VPU_DDR_BSS"))
        throw mv::ArgumentError(dm, "allocators", "VPU_DDR_BSS", "Computation model does not have VPU_DDR_BSS specified");

    if (cm.stageSize() == 0)
        throw mv::ArgumentError(cm , "stages count", "0", "Computation model does not have stages specified");

    mv::OpModel om(dm);
    auto stageIt = cm.getStage(0);

    auto sortedOps = om.topologicalSort();

    for (auto opIterator: sortedOps)
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

        else if (opType == "Constant" || opType == "ConstantInt" || opType == "ConstantDataElement" || opIterator->hasAttr("ImplicitFlow"))
        {
            pass.log(mv::Logger::MessageType::Debug, "Skipping allocation of opType " + opType);
            continue;
        }
        /*else if (opType == "ImplicitConcat")
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
        }*/
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
                       allocateUnpopulatedTensor(pass,dm,stageIt,inTensor);
                }
            }
            for (unsigned x = 0; x < opIterator->outputSlots(); ++x)
            {

                auto outTensor = opIterator->getOutputTensor(x);
                if (
                    (! outTensor->hasAttr("allocators")) &&
                    (! outTensor->hasAttr("modelInput") || ! outTensor->get<bool>("modelInput")) &&
                    (! outTensor->hasAttr("modelOutput") || ! outTensor->get<bool>("modelOutput"))
                    )
                {
                    allocateUnpopulatedTensor(pass,dm,stageIt,outTensor);
                }
            }
        }
    }
 }

//TODO::temporal hack, since I cannot actually get the allocator string from
//a tensor, since it may have multiple loations at one time.
//This pass only ensures allocators based on logical location decided in the
//passes until now

static std::map<std::string,std::string> location2Allocator =
{
        { "UPACMX", "VPU_CMX_UPA"},
        { "NNCMX", "VPU_CMX_NN" },
        { "DDR", "VPU_DDR_Heap"},
        { "INPUT", "ProgrammableInput"},
        { "OUTPUT", "ProgrammableOutput"},
        { "DEFAULT", "VPU_DDR_BSS"},
        { "BLOB", "GraphFile"}
};

void allocateImplicitOperationsKmbFcn(const mv::pass::PassEntry& pass,
                                            mv::ComputationModel& model,
                                            mv::TargetDescriptor&,
                                            mv::Element&,
                                            mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Allocating implicit tensors");

    mv::ControlModel cm(model);
    mv::DataModel dm(model);
    mv::OpModel om(dm);
    auto stageIt = cm.getStage(0);

    auto sortedOps = om.topologicalSort();

    for (auto opIterator: sortedOps)
    {

        if(!opIterator->hasAttr("ImplicitFlow"))
            continue;
        std::string opType = opIterator->getOpType();
        auto implicitFlow = opIterator->get<mv::ImplicitFlow>("ImplicitFlow");
        if ( implicitFlow.isImplicit() )
        {

            //TODO:: this every stage should declare in an abstract way the compositional logic
            //  so this phase can use it in an abstract manner... Currently will check manually
            //  for Concat and Slice because of happy days........
            //  basically need to get from tha leyer attributes some shcema of how and what coordinates to
            //  put buffer into eg  axis: 'W" ->[0-10][10-20][30-40] etc,....

            if(opType == "Concat" || opType == "ImplicitConcat")
            //this means that the Input tensors should be in the Output Tensor
            {
                auto outputTensor = opIterator->getOutputTensor(0);
                auto outputLocation = outputTensor->get<mv::Tensor::MemoryLocation>("Location");

                mv::Data::BufferIterator outputBuffer;

                if( !outputTensor->hasAttr("allocators"))
                {
                    pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() +
                            " Has no allocator. Will attempt to allocate based on logical location");
                    outputBuffer = allocateUnpopulatedTensor(pass,dm,stageIt,outputTensor);
                }
                else
                {
                    outputBuffer = dm.getBuffer(location2Allocator[outputLocation.toString()],stageIt,outputTensor);
                }

                auto inputSlots = opIterator->inputSlots();

                auto axis = mv::Shape::getAxis(opIterator->get<std::string>("axis"));
                std::vector<unsigned> running_concat_offset_LHS;
                std::vector<unsigned> running_concat_offset_RHS;

                auto prev_offset = 0;
                auto offset = 0;

                for(unsigned i = 0; i < inputSlots; i++)
                {
                    running_concat_offset_LHS.push_back(prev_offset + offset);
                    prev_offset = prev_offset + offset;
                    // Calculate for next tensor
                    offset = opIterator->getInputTensor(i)->getShape()[axis];
                    running_concat_offset_RHS.push_back(outputTensor->getShape()[axis] - prev_offset - offset);
                }

                for(unsigned i = 0; i != inputSlots; i++)
                {
                    auto inputTensor = opIterator->getInputTensor(i);
                    auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");
                    mv::Data::BufferIterator inputBuffer;

                    // If already allocated from a previous pass, deallocate.
                    // Note: This is probably not a good long term solution as we may have
                    // requirements from two different connections, this approach only resolves one.
                    // Probably restrictions on a tensor should be attributes of that tensor.
                    if (!inputTensor->hasAttr("allocators"))
                    {    inputBuffer = allocateUnpopulatedTensor(pass,dm,stageIt,inputTensor);
                        pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() + ""
                                " Has no allocator. Will attempt to allocate based on logical location");
                    }
                    else
                    {
                        inputBuffer = dm.getBuffer(location2Allocator[inputLocation.toString()],stageIt,inputTensor);
                    }

                    std::vector<std::size_t> lhs_padding(inputTensor->getShape().ndims());
                    std::vector<std::size_t> rhs_padding(inputTensor->getShape().ndims());


                    // This code assumes all tensors are of equal size. TODO: Assertions
                    auto lhs = running_concat_offset_LHS[i];
                    auto rhs = running_concat_offset_RHS[i];

                    lhs_padding.at(axis) = lhs;
                    rhs_padding.at(axis) = rhs;

                    auto NewBuffer = dm.moveTensor(location2Allocator[inputLocation.toString()],
                                                    inputBuffer, outputBuffer,
                                                    lhs_padding, rhs_padding);
                }
            }
            else if(opType == "ImplicitUnion")
            {
                //In implicit union case, we dont really have 1 master buffer, we are just using it
                // to have one output for the whole network, so each input to this op will still have
                // it's own buffer. We create the buffers but do NOT move them like in the other case
                // no slave/master case.
                auto outputTensor = opIterator->getOutputTensor(0);
                auto outputLocation = outputTensor->get<mv::Tensor::MemoryLocation>("Location");

                mv::Data::BufferIterator outputBuffer;

                if( !outputTensor->hasAttr("allocators"))
                {
                    pass.log(mv::Logger::MessageType::Warning, "Tensor " + outputTensor->getName() +
                            " Has no allocator. Will attempt to allocate based on logical location");
                    outputBuffer = allocateUnpopulatedTensor(pass,dm,stageIt,outputTensor);
                }
                else
                {
                    outputBuffer = dm.getBuffer(location2Allocator[outputLocation.toString()],stageIt,outputTensor);
                }


            }
            else if (opType == "Slice" || opType == "Crop")
            {
                auto outputTensor = opIterator->getOutputTensor(0);
                auto inputTensor = opIterator->getInputTensor(0);
                auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");
                auto outputLocation = outputTensor->get<mv::Tensor::MemoryLocation>("Location");

                mv::Data::BufferIterator inputBuffer;
                mv::Data::BufferIterator outputBuffer;

                if (!inputTensor->hasAttr("allocators"))
                {
                    inputBuffer = allocateUnpopulatedTensor(pass, dm, stageIt, inputTensor);
                    pass.log(mv::Logger::MessageType::Debug, "Tensor " + inputTensor->getName() + ""
                            " Has no allocator. Will attempt to allocate based on logical location");
                }
                else
                {
                    inputBuffer = dm.getBuffer(location2Allocator[inputLocation.toString()],stageIt,inputTensor);
                }

                if(!outputTensor->hasAttr("allocators"))
                {
                    pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() +
                            " Has no allocator. Will attempt to allocate based on logical location");
                    outputBuffer = allocateUnpopulatedTensor(pass, dm, stageIt, outputTensor);
                }
                else
                {
                    outputBuffer = dm.getBuffer(location2Allocator[outputLocation.toString()],
                                                stageIt,outputTensor);
                }

                auto ndims = inputTensor->getShape().ndims();
                auto inputShape = inputTensor->getShape();
                mv::Shape begin;
                mv::Shape size;
                if (opType == "Slice")
                {
                    begin = opIterator->get<mv::Shape>("begin");
                    size = opIterator->get<mv::Shape>("size");
                }
                else
                {
                    begin = {0,0,0,0};
                    size = outputTensor->getShape();
                }

                std::vector<std::size_t> lhs_padding(ndims);
                std::vector<std::size_t> rhs_padding(ndims);

                for(unsigned i = 0; i < ndims; i++)
                {
                    lhs_padding[i] = begin[i];
                    rhs_padding[i] = inputShape[i] - (begin[i] + size[i]);
                }

                auto NewBuffer = dm.moveTensor(location2Allocator[inputLocation.toString()],
                                                outputBuffer, inputBuffer,
                                                lhs_padding, rhs_padding);
            }
            else if (opType == "Copy" || opType == "Align" || opType == "ImplicitOutput")
            {
                auto outputTensor = opIterator->getOutputTensor(0);
                auto inputTensor = opIterator->getInputTensor(0);
                auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");
                auto outputLocation = outputTensor->get<mv::Tensor::MemoryLocation>("Location");
                mv::Data::BufferIterator inputBuffer;
                mv::Data::BufferIterator outputBuffer;

                if (!inputTensor->hasAttr("allocators"))
                {
                    inputBuffer = allocateUnpopulatedTensor(pass, dm, stageIt, inputTensor);
                    pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() + ""
                            " Has no allocator. Will attempt to allocate based on logical location");
                }
                else
                {
                    inputBuffer = dm.getBuffer(location2Allocator[inputLocation.toString()],
                                                stageIt, inputTensor);
                }

                if( !outputTensor->hasAttr("allocators"))
                {
                    pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() +
                            " Has no allocator. Will attempt to allocate based on logical location");
                    outputBuffer = allocateUnpopulatedTensor(pass, dm, stageIt, outputTensor);
                }
                else
                {
                    outputBuffer = dm.getBuffer(location2Allocator[outputLocation.toString()],
                                                stageIt, outputTensor);
                }

                if (opType == "Align")
                {
                    auto ndims = inputTensor->getShape().ndims();

                    std::vector<std::size_t> lhs_padding(ndims);
                    std::vector<std::size_t> rhs_padding(ndims);

                    mv::Shape begin = {0,0,0,0};
                    mv::Shape size = inputTensor->getShape();;
                    mv::Shape outputShape = outputTensor->getShape();
                    for(unsigned i = 0; i < ndims; i++)
                    {
                        lhs_padding[i] = begin[i];
                        rhs_padding[i] = outputShape[i] - (begin[i] + size[i]);
                    }
                    auto newBuffer = dm.moveTensor(location2Allocator[inputLocation.toString()],
                                                    inputBuffer, outputBuffer,
                                                    lhs_padding, rhs_padding);
                }
                else
                {
                    auto newBuffer = dm.moveTensor(location2Allocator[inputLocation.toString()],
                                                    inputBuffer, outputBuffer,
                                                    {0,0,0,0}, {0,0,0,0});
                }

            }
            else if((opType == "ImplicitReshape") || (opType == "ImplicitPermute"))
            {
                auto outputTensor = opIterator->getOutputTensor(0);
                auto inputTensor = opIterator->getInputTensor(0);
                auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");
                auto outputLocation = outputTensor->get<mv::Tensor::MemoryLocation>("Location");
                mv::Data::BufferIterator inputBuffer;
                mv::Data::BufferIterator outputBuffer;

                if (!inputTensor->hasAttr("allocators"))
                {
                    inputBuffer = allocateUnpopulatedTensor(pass, dm, stageIt, inputTensor);
                    pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() + ""
                            " Has no allocator. Will attempt to allocate based on logical location");
                }
                else
                {
                    inputBuffer = dm.getBuffer(location2Allocator[inputLocation.toString()],
                                                stageIt, inputTensor);
                }

                if( !outputTensor->hasAttr("allocators"))
                {
                    pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() +
                            " Has no allocator. Will attempt to allocate based on logical location");
                    outputBuffer = allocateUnpopulatedTensor(pass, dm, stageIt, outputTensor);
                }
                else
                {
                    outputBuffer = dm.getBuffer(location2Allocator[outputLocation.toString()],
                                                stageIt, outputTensor);
                }

                auto newBuffer = dm.moveTensor(location2Allocator[inputLocation.toString()],
                                                inputBuffer, outputBuffer,
                                                {0,0,0,0}, {0,0,0,0});

            }
            else
            {
                auto outputTensor = opIterator->getOutputTensor(0);
                pass.log(mv::Logger::MessageType::Debug, "Tensor " + outputTensor->getName() +
                            " has implicit flow but was not assigned an allocator.");
            }
            
        }
    }
}
