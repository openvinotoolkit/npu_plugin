#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void updateImplicitLayersQuantizationParamsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateImplicitLayersLocationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(UpdateImplicitLayersQuantizationParams)
            .setFunc(updateImplicitLayersQuantizationParamsFcn)
            .setDescription(
                "Update Quantization Params for Implicit Layers output after input layers have been quantized");
    }
    namespace pass
    {
        MV_REGISTER_PASS(UpdateImplicitLayersLocation)
            .setFunc(updateImplicitLayersLocationParamsFcn)
            .setDescription(
                "Update Location of Implicit Ops");
    }
}

void updateImplicitLayersQuantizationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto sortedOps = om.topologicalSort();
    for(auto opIt : sortedOps)
    {
        if (!opIt->isImplicit() || opIt->getOpType() == "ImplicitInput" || opIt->getOpType() == "ImplicitInputSlice")
            continue;

        if (opIt->getOpType() == "ImplicitConcat" || opIt->getOpType() == "Concat")
        {
            auto inputs = opIt->getInputTensor();
            auto output = opIt->getOutputTensor(0);

            if (!inputs[0]->hasAttr("quantParams"))
                continue;

            // Initilize concat output quant parameters with first input
            mv::QuantizationParams outputQuantParamsFull = inputs[0]->getQuantParams();

            // If concatenation is along channel axis quantization parameters need to be also
            // concatenated if they are per channel
            if (opIt->get<std::string>("axis") == "C")
            {
                auto output_zp_vec = outputQuantParamsFull.getZeroPoint();
                auto output_scale_vec = outputQuantParamsFull.getScale();
                auto output_min_vec = outputQuantParamsFull.getMin();
                auto output_max_vec = outputQuantParamsFull.getMax();
                std::vector<unsigned> output_shift_vec;
                std::vector<unsigned> output_mult_vec;
                if (outputQuantParamsFull.hasAttr("shift") && outputQuantParamsFull.hasAttr("mult"))
                {
                    output_shift_vec = outputQuantParamsFull.getShift();
                    output_mult_vec = outputQuantParamsFull.getMult();
                }

                // Iterate through rest of inputs and concatenate quant parameters, but only
                // if they are per channel and not for whole tensor
                for (std::size_t idx = 1; idx < inputs.size(); idx++)
                {
                    mv::QuantizationParams inputQuantParams  = inputs[idx]->getQuantParams();
                    auto zp_vec = inputQuantParams.getZeroPoint();
                    auto scale_vec = inputQuantParams.getScale();
                    auto min_vec = inputQuantParams.getMin();
                    auto max_vec = inputQuantParams.getMax();

                    if(output_zp_vec.size() > 1)
                        output_zp_vec.insert(output_zp_vec.end(), zp_vec.begin(), zp_vec.end());
                    if(output_scale_vec.size() > 1)
                        output_scale_vec.insert(output_scale_vec.end(), scale_vec.begin(), scale_vec.end());
                    if(output_min_vec.size() > 1)
                        output_min_vec.insert(output_min_vec.end(), min_vec.begin(), min_vec.end());
                    if(output_max_vec.size() > 1)
                        output_max_vec.insert(output_max_vec.end(), max_vec.begin(), max_vec.end());

                    if (outputQuantParamsFull.hasAttr("shift") && outputQuantParamsFull.hasAttr("mult"))
                    {
                        auto shift_vec = inputQuantParams.getShift();
                        auto mult_vec = inputQuantParams.getMult();

                        if(output_shift_vec.size() > 1)
                            output_shift_vec.insert(output_shift_vec.end(), shift_vec.begin(), shift_vec.end());
                        if(output_mult_vec.size() > 1)
                            output_mult_vec.insert(output_mult_vec.end(), mult_vec.begin(), mult_vec.end());
                    }
                }

                // Build the whole quant params with new concatenated vectors
                if (outputQuantParamsFull.hasAttr("shift") && outputQuantParamsFull.hasAttr("mult"))
                    outputQuantParamsFull = mv::QuantizationParams(output_zp_vec, output_scale_vec, output_min_vec, output_max_vec, output_shift_vec, output_mult_vec);
                else
                    outputQuantParamsFull = mv::QuantizationParams(output_zp_vec, output_scale_vec, output_min_vec, output_max_vec);
            }

            output->setQuantParams(outputQuantParamsFull);
        }
        else if(opIt->getOpType() == "ImplicitSlice" || opIt->getOpType() == "Slice")
        {
            auto input = opIt->getInputTensor(0);
            if (!input->hasAttr("quantParams"))
                continue;

            auto output = opIt->getOutputTensor(0);
            auto inputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
            auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

            auto numOfSlices = inputChannels / outputChannels;

            mv::QuantizationParams inputQuantParams = input->get<mv::QuantizationParams>("quantParams");
            if (numOfSlices > 1)
            {
                // If slicing is done along channel axis, quantization parameters need to also be
                // correctly partitioned
                auto sliceId = (opIt->get<mv::Shape>("begin")[mv::IO_CHANNEL_DIMENSION])/outputChannels;
                output->set<mv::QuantizationParams>("quantParams", inputQuantParams.getSlice(sliceId, numOfSlices));
            }
            else
            {
                // Otherwise just populate quant params from input
                output->set<mv::QuantizationParams>("quantParams", inputQuantParams);
            }
        }
        else
        {
            auto input = opIt->getInputTensor(0);
            if (input->hasAttr("quantParams"))
            {
                auto output = opIt->getOutputTensor(0);
                mv::QuantizationParams &inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                output->set<mv::QuantizationParams>("quantParams", inputQuantization);
            }
        }
    }
}

std::vector<mv::Data::OpListIterator> findOutputNodeParentImplicitOps(mv::ComputationModel& model, const mv::Data::OpListIterator &op)
{
    mv::OpModel om(model);
    std::vector<mv::Data::OpListIterator> implicitOps;
    auto parentOp = om.getSourceOp(op->getInputTensor(0));
    
    while(parentOp->isImplicit())
    {
        implicitOps.push_back(parentOp); 
        parentOp = om.getSourceOp(parentOp->getInputTensor(0));
       
    }
    implicitOps.pop_back(); 
    return implicitOps;
}

// This is a helper function to recursively scan children of operation until it reaches DPU tasks.
// In case of a DPU task check input sparsity setting.
static void scanDpuForSparsityInputRecursiveDown(mv::Data::OpListIterator exploreOp, std::vector<bool>& sparsityEnVec, mv::OpModel& om)
{
    for(auto nextOp = exploreOp.leftmostChild(); nextOp != om.opEnd(); ++nextOp)
    {
        auto opType = nextOp->getOpType();
        if(opType == "DPUTask")
            sparsityEnVec.push_back(nextOp->hasAttr("inputActivationSparsity") ? nextOp->get<bool>("inputActivationSparsity") : false);
        else
            scanDpuForSparsityInputRecursiveDown(nextOp, sparsityEnVec, om);
    }
}

// This function is to check if with given tensor order and slicing parameters, tensor splitting
// can be performed in CMX or if it has to go through DDR (->Slice->DMA->DDR->DMA->DPU->).
// This check is performed because DPU IDU unit cannot directly load input_data if it is not
// a continguous block of memory - so when input tensor dimensions do not match input strides
// In the effect slicing in CMX as an input to DPU task can be done only along major dimension.
// This restriction is not relevant if sparsity support functionality is used for input of DPU task
bool isSliceToBeMovedToDdr(mv::Data::OpListIterator sliceOp, mv::OpModel& om)
{
    // Check if slice was already requested to be performed in DDR
    if (sliceOp->hasAttr("force_slice_in_DDR") && sliceOp->get<bool>("force_slice_in_DDR"))
        return true;

    auto inputTensor = sliceOp->getInputTensor(0);
    // All this checking is relevant if input is in CMX or already in DDR.
    // If it is INPUT or OUTPUT then it can return early as the code is intended to analyze
    // split execution between CMX and DDR
    if (inputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::INPUT ||
        inputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::OUTPUT)
        return false;

    // Check if slicing is performed along axis which with given input tensor order
    // can be done as splits into contiguous blocks of memory
    // NHWC - split along H
    // NCHW - split along C
    bool sliceAlongMajorDim = true;
    auto order = inputTensor->getOrder();
    auto shape = inputTensor->getShape();

    auto begin = sliceOp->get<mv::Shape>("begin");
    auto size = sliceOp->get<mv::Shape>("size");

    if (shape == size)
        return false;

    // Mask with information on which dimension with given tensor
    // order split will be supported by DPU IDU. Boolean value
    // idicates if certain dimension must match.
    //                             |  W   |  H   |  C   |  N  |
    bool allowed_mask_split_H[4] = {true,  false, true,  true};
    bool allowed_mask_split_C[4] = {true,  true,  false, true};
    bool *allowed_mask_split = NULL;

    if (order == mv::Order("NHWC"))
    {
        allowed_mask_split = allowed_mask_split_H;
    } else if (order == mv::Order("NCHW")) {
        allowed_mask_split = allowed_mask_split_C;
    }

    if (allowed_mask_split == NULL)
        return true;

    for (size_t i = 0; i < 4; i++)
    {
        if((size[i] == shape[i]) != allowed_mask_split[i])
        {
            sliceAlongMajorDim = false;
            break;
        }
    }

    if (sliceAlongMajorDim)
        return false;

    // If slice is not done along major dimension check if all subsequent DPU tasks
    // have enabled input sparsity
    std::vector<bool> sparsityEnVec;
    scanDpuForSparsityInputRecursiveDown(sliceOp, sparsityEnVec, om);
    auto sparsityEnableOnAllTasks = true;
    auto sparsityEnableOnAnyTasks = false;
    for (bool sparsityEn : sparsityEnVec)
    {
        sparsityEnableOnAllTasks &= sparsityEn;
        sparsityEnableOnAnyTasks |= sparsityEn;
    }
    sparsityEnableOnAllTasks &= sparsityEnableOnAnyTasks;

    // If in case of not doing the slice along major dimension but all subsequent tasks
    // have input sparsity enabled then slicing can be located in CMX
    if (sparsityEnableOnAllTasks)
        return false;

    // If sparsity is enabled only on some tasks signal this as unsupported config
    if (!sparsityEnableOnAllTasks && sparsityEnableOnAnyTasks)
        throw mv::AttributeError("isSliceToBeForcedInDdr", " slice op " + sliceOp->getName() +
                                            " has DPU childen tasks which do not have matching input sparsity enable state");

    return true;
}

void updateImplicitLayersLocationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto sortedOps = om.topologicalSort();
    for(auto opIt : sortedOps)
    {
         std::string opType = opIt->getOpType();

        if (opType == "Slice")
        {
            if (isSliceToBeMovedToDdr(opIt, om))
            {
                opIt->set<bool>("force_slice_in_DDR", true);
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::DDR);
            } else {
                auto inputMemoryLocation = opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", inputMemoryLocation);
            }
        }
        else if (opType == "Crop")
        {
            // Recursively search for non-implicit output op
            auto outputOp = opIt.leftmostOutput().sink();

            // Crop is correctly accounted for with a spilling DMA
            if (outputOp->getOpType() == "DMATask" &&
                outputOp->get<mv::DmaDirection>("direction") == mv::NNCMX2DDR)
                continue;

            while(outputOp->isImplicit())
            {
                outputOp = outputOp.leftmostOutput().sink();
            }

            auto outputOpMemoryLocation = outputOp->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto newMemoryLocation = (outputOpMemoryLocation == mv::Tensor::MemoryLocation::OUTPUT)
                    ? mv::Tensor::MemoryLocation::OUTPUT
                    : mv::Tensor::MemoryLocation::DDR;
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", newMemoryLocation);
        }
        //NOTE: Temporary handle for the scheduler in order to place the required DMA-s for the copy operation
        else if (opType == "Copy")
        {
            if (opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::NNCMX)
            {
                auto parentOp = om.getSourceOp(opIt->getInputTensor(0));
                //Sink Ops of the Input of the Copy layer
                // auto sinkOps = findSinkLayers(dm, opIt->getInputTensor(0));
                std::vector<mv::Data::OpListIterator> sinkOps;
                auto outputFlow = parentOp.leftmostOutput();
                std::size_t copyId = 0;
                std::size_t dpuId = 0;
                std::unordered_map<std::string, std::vector<mv::Data::FlowSiblingIterator>> tasks_flows;
                while (outputFlow != om.flowEnd())
                {
                    if (outputFlow.sink()->getOpType() == "Copy"
                        && outputFlow.sink()->getName() == opIt->getName()) // In case of parallel branches, just this guy
                    {
                        copyId = outputFlow->get<std::size_t>("sinkInput");
                        tasks_flows["Copy"].push_back(outputFlow);
                        sinkOps.push_back(outputFlow.sink());
                    }
                    else
                    {
                        auto sinkInput = outputFlow.sink().leftmostInput();
                        while(sinkInput != om.flowEnd()){
                            if(sinkInput.source()->getName() == opIt->getName()){
                                dpuId = outputFlow->get<std::size_t>("sinkInput");
                                tasks_flows["DPUTask"].push_back(outputFlow);
                                sinkOps.push_back(outputFlow.sink());
                            }
                            ++sinkInput;
                        }
                    }
                    ++outputFlow;
                }
                auto compensatorOutput = om.dMATask(opIt->getName() + "_copyDMA",
                                                    opIt->getInputTensor(0),
                                                    mv::DmaDirectionEnum::NNCMX2DDR,
                                                    0);
                compensatorOutput->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::DDR);
                om.getSourceOp(compensatorOutput)->set<unsigned>("opId", opIt->get<unsigned>("opId"));

                for (auto sinkOp:sinkOps)
                {
                    if (sinkOp->getOpType() == "Copy")
                    {
                        //NOTE: neutral Copy has only 0
                        sinkOp->setInputTensor(compensatorOutput,0, false);
                        om.undefineFlow(tasks_flows["Copy"][0]);
                        om.defineFlow(compensatorOutput, sinkOp, copyId);
                    }
                    else
                    {
                        sinkOp->setInputTensor(compensatorOutput, 0, false);
                        om.undefineFlow(tasks_flows["DPUTask"][0]);
                        om.defineFlow(compensatorOutput, sinkOp, dpuId);
                    }
                }

            }
        }
        else if (opType == "ImplicitReshape" || opType == "ImplicitPermute")
        {
            auto input = opIt->getInputTensor(0);
            // Recursively search for non-implicit input op
            auto inputOp = om.getSourceOp(input);
            while(inputOp->isImplicit())
            {
                input = inputOp->getInputTensor(0);
                inputOp = om.getSourceOp(input);
            }
            // Recursively search for non-implicit output op
            auto outputOp = opIt.leftmostOutput().sink();
            while(outputOp->isImplicit())
            {
                outputOp = outputOp.leftmostOutput().sink();
            }
            // If to/from DDR, pick DDR as location; else use inputTensor location
            auto inputOpMemoryLocation = inputOp->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto outputOpMemoryLocation = outputOp->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto newMemoryLocation = ((inputOpMemoryLocation == mv::Tensor::MemoryLocation::DDR) ||
                                       outputOpMemoryLocation == mv::Tensor::MemoryLocation::DDR)
                    ? mv::Tensor::MemoryLocation::DDR
                    : opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", newMemoryLocation);
        }

        // If the last operation is streamed and the concat is in DDR and followed by implicit Ops
        // Then you should do a DMA directly from NNCMX to Programamble Output
        else if (opType == "Output" && om.getSourceOp(opIt->getInputTensor(0))->isImplicit())
        {
            auto outputNodeParentImplicitOps = findOutputNodeParentImplicitOps(om, opIt);
            if(outputNodeParentImplicitOps.size())
            {
                for(auto const& implicitOp : outputNodeParentImplicitOps)
                {
                    for( size_t input = 0; input < implicitOp->inputSlots(); input++)
                        implicitOp->getInputTensor(input)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
            
                    for( size_t output = 0; output < implicitOp->outputSlots(); output++)
                        implicitOp->getOutputTensor(output)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
                }
            }
        }
        else if (opType == "ImplicitInput")
        {
            auto inputMemoryLocation = opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", inputMemoryLocation);
        }
    }
}
