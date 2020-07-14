#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include <cmath>
#include <math.h>

static const std::size_t WT_ELEMENTS_PER_CHANNEL = 4;
static const std::size_t ALU_HALT_OPCODE = 6;
static const std::size_t ALU_LOAD = 2;
static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void generateInstructionListTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeBiasTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateStorageElementPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateInstructionListTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );
        MV_REGISTER_PASS(GenerateInstructionListTables)
        .setFunc(generateInstructionListTablesFcn)
        .setDescription(
            "Generates instruction list tables for the Tasks that need them"
        );
        MV_REGISTER_PASS(PopulateWeightsTablesPointers)
        .setFunc(populateWeightsTablesPointersFcn)
        .setDescription(
            "Populate WeightsTables"
        );
        MV_REGISTER_PASS(PopulateWeightsTablesQuantization)
        .setFunc(populateWeightsTablesQuantizationFcn)
        .setDescription(
            "Populate WeightsTables"
        );
        MV_REGISTER_PASS(RemoveBiasTensors)
        .setFunc(removeBiasTensorsFcn)
        .setDescription(
            "remove bias tensors after been adding to all weight tables"
        );
        MV_REGISTER_PASS(PopulateStorageElementPointers)
        .setFunc(populateStorageElementPointersFcn)
        .setDescription(
            "Populate storage element maps for activations"
        );
        MV_REGISTER_PASS(PopulateInstructionListTables)
        .setFunc(populateInstructionListTablesFcn)
        .setDescription(
            "Populate instruction list tables for PWL solution"
        );
    }
}

void populatePointerMultiCluster(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, const std::vector<int64_t>& increments, long int offset, std::size_t addingIndex, mv::ComputationModel &model, mv::Data::TensorIterator tensor)
{
    //std::cout << "Populating " << std::to_string(addingIndex) << " pointer of weights table for op " << dpuTaskOp->getName() << std::endl;
    if (dpuTaskOp->get<std::string>("splitStrategy") != "SplitOverK")
    {
        for (size_t i = 0, k = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL, ++k)
            weightsTableData->at(i + addingIndex) = offset + increments[k];
    }
    else
    {
        // Saving a backup copy of the offset
        long int new_offset = offset;

        unsigned numClusters =  model.getGlobalConfigParams()->get<int>("Number_of_Clusters");
        std::size_t sizeToIterate = 0;
        std::size_t totalSizeToIterate = 0;
        std::size_t k = 0;
        bool isSparse = tensor->isSparse();
        for (unsigned i = 0; i < numClusters; i++)
        {
            // Resetting offset at the beginning of the cluster
            offset = new_offset;
            // Resetting k index only when weights are not sparse
            if(!isSparse)
                k = 0;

            // Filling cluster
            for (size_t j = 0; j < weightsTableData->getSubTensor(i).size(); j+=WT_ELEMENTS_PER_CHANNEL)
                weightsTableData->at(j + addingIndex + totalSizeToIterate) = offset + increments[k++];

            // Preparing for next iteration
            sizeToIterate = tensor->getSubTensor(i).getShape()[mv::KERNEL_OUTPUT_CHANNELS] * WT_ELEMENTS_PER_CHANNEL;
            totalSizeToIterate += sizeToIterate;
        }
    }
    return;
}

void populateWeightsTablesDataPointers(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    auto taskOp = dpuTaskOp->get<std::string>("taskOp");

    // Max pooling does not need DataPointer
    // Eltwise doesn't have weights table at all
    if(taskOp == "Conv" || taskOp == "ChannelMajorConvolution" ||
       taskOp == "DepthwiseConv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);

        // NOTE: Output channels taken from the weights, hence already padded
        auto outputChannels = weights->getShape()[mv::KERNEL_OUTPUT_CHANNELS];
        auto strategy = dpuTaskOp->get<std::string>("splitStrategy");

        long int offset = weights->getAddress();
        std::vector<int64_t> increments;

        if(weights->isSparse())
        {
            if(strategy != "SplitOverK")
                increments = weights->getKernelDataOffsets();
            else
            {
                unsigned numClusters =  model.getGlobalConfigParams()->get<int>("Number_of_Clusters");
                for(std::size_t i = 0; i < numClusters; ++i)
                {
                    auto kernelOffsets = weights->getSubTensor(i).getKernelDataOffsets();
                    increments.insert(increments.end(), kernelOffsets.begin(), kernelOffsets.end());
                }
            }
        }
        else
        {
            long int increment = weights->getShape()[mv::KERNEL_WEIGHT_SETS] * (weights->getDType().getSizeInBits() / 8);
            increments = std::vector<int64_t>(outputChannels, 0);
            for(unsigned i = 1; i < outputChannels; ++i)
                increments[i] = increments[i-1] + increment;
        }
        populatePointerMultiCluster(weightsTableData, dpuTaskOp, increments, offset, 0, model, weights);
    }
}



void populateWeightsTablesSparsityPointers(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::DataModel dm(model);

    auto taskOp = dpuTaskOp->get<std::string>("taskOp");
    if(taskOp == "Conv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);

        // NOTE: Output channels taken from the weights, hence already padded
        auto outputChannels = weights->getShape()[mv::KERNEL_OUTPUT_CHANNELS];
        if(weights->isSparse())
        {
            auto weightsSparsityMap = dm.getTensor(weights->getSparsityMap()->getName());
            long int offset = weightsSparsityMap->getAddress();
            auto sparsityMapSizeInWords = weightsSparsityMap->getShape().totalSize();
            auto sparsityMapSizeInBytes = sparsityMapSizeInWords * weightsSparsityMap->getDType().getSizeInBits() / 8;
            long int increment = sparsityMapSizeInBytes / outputChannels;

            std::vector<int64_t> increments = std::vector<int64_t>(outputChannels, 0);
            for(unsigned i = 1; i < outputChannels; ++i)
                increments[i] = increments[i-1] + increment;

            populatePointerMultiCluster(weightsTableData, dpuTaskOp, increments, offset, 1, model, weightsSparsityMap);
        }
        else
        {
            // Dense ZMajor Convolution case
            // Not using the generic function because it's a super simple case
            long int offset = 0xFFFFFF; // NOTE: Implementation defined
            for (size_t i = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL)
                  weightsTableData->at(i+1) = offset;
        }
    }
    else if(taskOp == "DepthwiseConv"  ||
            taskOp == "ChannelMajorConvolution" ||
            taskOp == "MaxPool")
    {
        // We have fake sparsity here! Yuppi!
        auto activationWindow = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("fakeSparsityIndex"));
        auto activationWindowSizeInWords = activationWindow->getShape().totalSize();
        auto activationWindowSizeInBytes = activationWindowSizeInWords * activationWindow->getDType().getSizeInBits() / 8;

        auto outputChannels = dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

        auto paddedOutputChannels = outputChannels;
        auto globalParams = model.getGlobalConfigParams();
        int pad = globalParams->hasAttr("VPU2ChannelPadding") ? globalParams->get<int>("VPU2ChannelPadding") : 16;

        paddedOutputChannels = mv::round_up(outputChannels, pad);
        auto increment = activationWindowSizeInBytes / paddedOutputChannels;

        long int offset = activationWindow->getAddress();
        std::vector<int64_t> increments = std::vector<int64_t>(paddedOutputChannels, 0);
        for(unsigned i = 1; i < paddedOutputChannels; ++i)
            increments[i] = increments[i-1] + increment;

        populatePointerMultiCluster(weightsTableData, dpuTaskOp, increments, offset, 1, model, activationWindow);
    }
}


void populateWeightsTablesActivationAndBias(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::DataModel dm(model);
    mv::QuantizationParams quantParams = {{},{},{},{}};
    auto output = dpuTaskOp->getOutputTensor(0);
    auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto paddedOutputChannels = outputChannels;
    auto globalParams = model.getGlobalConfigParams();
    int pad = globalParams->hasAttr("VPU2ChannelPadding") ? globalParams->get<int>("VPU2ChannelPadding") : 16;

    paddedOutputChannels = mv::round_up(outputChannels, pad);

    std::vector<int32_t> mScaled(paddedOutputChannels, 0);
    std::vector<int32_t> mShift(paddedOutputChannels, 0);
    if(output->hasAttr("quantParams"))
    {
        if (dpuTaskOp->hasAttr("pwlQuantParams"))
            quantParams = dpuTaskOp->get<mv::QuantizationParams>("pwlQuantParams");
        else
            quantParams = dpuTaskOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        if (!quantParams.isEmpty())
        {
            auto mult = quantParams.getMult();
            auto shift = quantParams.getShift();
            std::transform(mScaled.begin(), mScaled.end(), mult.begin(), mScaled.begin(), std::plus<int32_t>());
            std::transform(mShift.begin(), mShift.end(), shift.begin(), mShift.begin(), std::plus<int32_t>());
            for (size_t idx = outputChannels; idx < paddedOutputChannels; idx++)
            {
                mScaled[idx] = mScaled[0];
                mShift[idx] = mShift[0];
            }
        }
    }
    std::vector<mv::DataElement> biasData;
    bool hasBias = dpuTaskOp->hasAttr("bias");
    bool hasPPETask = dpuTaskOp->hasAttr("PPETask");

    mv::Data::TensorIterator bias;
    if (hasBias)
    {
        bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
        biasData = bias->getData(); //Bias has the type Int32 in both cases above
    }

    unsigned round_mode = 1;
    std::vector<int32_t> round32(outputChannels, round_mode);
    std::vector<int32_t> reluMultData(outputChannels, 0);
    if (hasPPETask)
    {
        auto ppeFF = dpuTaskOp->get<mv::PPETask>("PPETask").getFixedFunction();
        auto& ppeLayers = ppeFF.getLayers();
        auto isLRelu = std::find(ppeLayers.begin(), ppeLayers.end(), mv::PPELayerTypeEnum::PPELayerType_LPRELU) != ppeLayers.end();
        if (isLRelu)
            std::fill(reluMultData.begin(), reluMultData.end(), dpuTaskOp->get<mv::PPETask>("PPETask").getFixedFunction().getLReluMult());
    }

    for (size_t i = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL)
    {
        weightsTableData->at(i+2) = static_cast<long int>((mScaled[i/WT_ELEMENTS_PER_CHANNEL] << 16) | (round32[i/WT_ELEMENTS_PER_CHANNEL] << 14) | (mShift[i/WT_ELEMENTS_PER_CHANNEL]) << 8) | reluMultData[i/WT_ELEMENTS_PER_CHANNEL];
        if (hasBias)
            weightsTableData->at(i+3) = biasData[i/WT_ELEMENTS_PER_CHANNEL];
    }
}
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("weightsTableIndex"));
                populateWeightsTablesActivationAndBias(weightsTable, dpuTaskOp, model);
            }
        }
    }
}
static void removeBiasTensorsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::set<std::string> biasNamesToRemove;
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                bool hasBias = dpuTaskOp->hasAttr("bias");
                if (hasBias)
                {
                    auto bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
                    biasNamesToRemove.insert(bias->getName());
                    dpuTaskOp->erase("bias");
                }
            }
        }
    }
    for(auto biasName = biasNamesToRemove.begin(); biasName != biasNamesToRemove.end(); ++biasName)
    {
        auto bias = dm.getTensor(*biasName);
        dm.undefineTensor(bias);
    }
}

static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("weightsTableIndex"));
                populateWeightsTablesDataPointers(weightsTable, dpuTaskOp, model);
                populateWeightsTablesSparsityPointers(weightsTable, dpuTaskOp, model);
            }
        }
    }
}

void populateActivationStorageElementMap(
    mv::Data::OpListIterator op,
    mv::ComputationModel& model)
{

    using clusterSolverFunc =
        std::function<mv::Tensor*(
        mv::Data::OpListIterator, size_t, size_t)>;

    using displacementCalcFunc =
        std::function<std::pair<long int, long int>(
        mv::Data::OpListIterator, size_t, clusterSolverFunc, size_t)>;

    const std::vector<clusterSolverFunc> clusterSolversFunctors = {
        [](mv::Data::OpListIterator op, size_t tidx, size_t clidx)
        {
            return &*op->getInputTensor(tidx);
        },
        [](mv::Data::OpListIterator op, size_t tidx, size_t clidx)
        {
            return &op->getInputTensor(tidx)->getSubTensor(clidx);
        }
    };

    const std::unordered_map<std::string, displacementCalcFunc> displacementFunctors =
    {
        {
            "Conv",
            [](mv::Data::OpListIterator op,
                size_t inputTensorIdx,
                clusterSolverFunc clSolver,
                size_t clIdx){
                std::vector<std::pair<long int, long int>> displacements;
                auto offset = 0;
                auto increment =
                    clSolver(op, inputTensorIdx, clIdx)->getShape()[mv::IO_CHANNEL_DIMENSION] *
                    (clSolver(op, inputTensorIdx, clIdx)->getDType().getSizeInBytes());
                return std::make_pair(offset, increment);
            }
        },
        {
            "Eltwise",
            [](mv::Data::OpListIterator op,
                size_t inputTensorIdx,
                clusterSolverFunc clSolver,
                size_t clIdx){
                auto base_addr =std::min(
                    clSolver(op, 0, clIdx)->getAddress(),
                    clSolver(op, 1, clIdx)->getAddress());
                auto offset = clSolver(op, inputTensorIdx, clIdx)->getAddress() - base_addr;
                auto increment =
                    clSolver(op, inputTensorIdx, clIdx)->getShape()[mv::IO_CHANNEL_DIMENSION] *
                    (clSolver(op, inputTensorIdx, clIdx)->getDType().getSizeInBytes());
                return std::make_pair(offset, increment);
            }
        }
    };

    const std::vector<std::string> segmentableStrategies = {"SplitOverH", "HKSwitch"};
    auto dispFunctor = displacementFunctors.at(op->get<std::string>("taskOp"));

    auto inputTensorIdx = 0;
    for (auto tidx : op->get<std::vector<std::size_t>>("storageElementIndex"))
    {
        auto storageElementTable = op->getInputTensor(tidx);
        std::vector<int64_t> table_offsets(storageElementTable->getShape().totalSize(), 0);

        if (std::find(segmentableStrategies.cbegin(), segmentableStrategies.cend(),
            op->get<std::string>("splitStrategy")) == segmentableStrategies.cend()) {
                auto disp = dispFunctor(op, inputTensorIdx, clusterSolversFunctors[0], 0);
                for (size_t i = 0; i < table_offsets.size(); ++i)
                    table_offsets[i] =
                        ((disp.first + i * disp.second) <<
                        SHIFT_FOR_STORAGE_ELEMENT);
        }
        else
        {
            auto numClusters =
                model.getGlobalConfigParams()->get<int>("Number_of_Clusters");
            auto running_index = 0;

            for (size_t cl = 0; cl < numClusters; cl++) {
                auto disp = dispFunctor(op, inputTensorIdx, clusterSolversFunctors[1], cl);
                auto clTotalSize =
                    storageElementTable->getSubTensor(cl).getShape().totalSize();
                for (size_t i = 0; i < clTotalSize; ++i)
                    table_offsets[running_index + i] =
                        ((disp.first + i * disp.second) <<
                        SHIFT_FOR_STORAGE_ELEMENT) + cl;
                running_index += clTotalSize;
            }
        }
        storageElementTable->populate(table_offsets, mv::Order("NHWC"));
        inputTensorIdx++;
    }
}

// Sub function to generate storage element pointer for dilated convolution

void populateActivationStorageElementMapForDilatedConvolution(mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    auto input = dpuTaskOp->getInputTensor(0);
    auto subConvIndex = dpuTaskOp->get<unsigned>("subConvIndex");
    auto activationStorageElement = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::vector<std::size_t>>("storageElementIndex")[0]);
    auto dilationFactor = dpuTaskOp->get<unsigned>("originalDilationFactor");
    auto originalWidth = dpuTaskOp->get<mv::Shape>("originalShape")[mv::IO_WIDTH_DIMENSION];
    auto inputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto width = activationStorageElement->getShape()[mv::IO_WIDTH_DIMENSION];
    auto height = activationStorageElement->getShape()[mv::IO_HEIGHT_DIMENSION];

    std::vector<int64_t> unpopulated_offsets(width*height, 0);
    unsigned subConvRowIdx = subConvIndex/dilationFactor;
    unsigned subConvColIdx = subConvIndex%dilationFactor;
    long int increment = inputChannels * (input->getDType().getSizeInBits() / 8) ;

    long int subConvElementIncrement = increment * dilationFactor;
    long int subConvRowIncrement = increment * originalWidth * dilationFactor;
    long int subConvOffset = increment * subConvColIdx + subConvRowIdx*originalWidth*increment;

    unsigned i = 0;
    unsigned rowOffset = subConvOffset;
    for(unsigned h = 0; h < height; ++h)
    {
        for(unsigned w = 0; w < width; ++w)
        {
            unpopulated_offsets[i++] = ((rowOffset + w * subConvElementIncrement ) << SHIFT_FOR_STORAGE_ELEMENT);
        }
        rowOffset += subConvRowIncrement;
    }
    activationStorageElement->populate(unpopulated_offsets, mv::Order("NHWC"));
}

int64_t getSmallestInputAddress(mv::Data::OpListIterator implicitJoin)
{
    auto numberInputs = implicitJoin.inputsSize();
    auto minBaseAddress = implicitJoin->getInputTensor(0)->getAddress();
    for (size_t i=1; i < numberInputs; i++)
    {
        auto address = implicitJoin->getInputTensor(i)->getAddress();
        if (address < minBaseAddress)
            minBaseAddress = address;
    }

    //std::cout << " minBaseAddress " << std::hex << minBaseAddress << std::endl;
    return minBaseAddress;
}

void populateActivationStorageElementMapForLayerAfterDilatedConvolution(mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);

    auto input = dpuTaskOp->getInputTensor()[0];
    auto parentImplicitOp = om.getSourceOp(input);
    while (parentImplicitOp->getOpType() != "ImplicitJoin")
    {
        parentImplicitOp = om.getSourceOp(parentImplicitOp->getInputTensor()[0]);
    }
    std::size_t numberSubConvs = 0;
    int64_t inputBaseAddress = 0;
    auto activationStorageElement = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::vector<std::size_t>>("storageElementIndex")[0]);
    auto width = activationStorageElement->getShape()[mv::IO_WIDTH_DIMENSION];
    auto height = activationStorageElement->getShape()[mv::IO_HEIGHT_DIMENSION];
    std::vector<int64_t> unpopulated_offsets(width*height, 0);
    auto inputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
    long int increment = inputChannels * (input->getDType().getSizeInBits() / 8) ;

    //NOTE: The code referring to the previous operation if concat
    //is redundant as the final implementation was not to use an adittional operation
    //but resolve the unshuffling of the tensor through the 3D-DMAs, so I am leaving to comments


//    if (parentImplicitOp->getOpType() == "ImplicitJoin")
//    {
    numberSubConvs = parentImplicitOp.inputsSize();
    inputBaseAddress = getSmallestInputAddress(parentImplicitOp);
    //Original DF factor is sqrt() of inputs to ImplicitJoin
    unsigned int originalDilationFactor = std::sqrt(numberSubConvs);

    //for simplicity we pick base address as the smallest of all subconvs output addresses (to avoid negatives)
    unsigned i = 0;
    for(unsigned h = 0; h < height; ++h)
    {
        unsigned subConvRowIdx = (h%originalDilationFactor)*originalDilationFactor;
        for(unsigned w = 0; w < width; ++w)
        {
            //get base address based on subConvIdx
            unsigned subConvIdx = subConvRowIdx + w%originalDilationFactor;
            auto subConvBaseAddressOffset = parentImplicitOp->getInputTensor(subConvIdx)->getAddress() - inputBaseAddress;
            auto subConvWidth = parentImplicitOp->getInputTensor(subConvIdx)->getShape()[mv::IO_WIDTH_DIMENSION];
            //calc offset from start of subconv
            unsigned subConvElementIdx = (h/originalDilationFactor)*subConvWidth + (w/originalDilationFactor);
            unsigned subConvElementOffset = subConvElementIdx * increment;

            unpopulated_offsets[i++] = ((subConvBaseAddressOffset + subConvElementOffset) << SHIFT_FOR_STORAGE_ELEMENT);
            //std::cout << " row " << h << " col " << w << " address "  <<  std::hex << unpopulated_offsets[i-1] << " not shifted " << (subConvBaseAddressOffset + subConvElementOffset) << std::endl;
        }
    }
//    }
//    else if (parentImplicitOp->getOpType() == "DMATask" &&
//             om.getSourceOp(parentImplicitOp->getInputTensor()[0])->getOpType() == "ImplicitConcat" &&
//             om.getSourceOp(parentImplicitOp->getInputTensor()[0])->get<bool>("joinSimulation"))
//    {
//        numberSubConvs = om.getSourceOp(parentImplicitOp->getInputTensor()[0])->get<size_t>("dilationSubConvs");
//        unsigned int originalDilationFactor = std::sqrt(numberSubConvs);
//        unsigned i = 0;
//        unsigned subConvHeight = ceil((double)height / originalDilationFactor); //height of bigger subconvs
//        unsigned subConvWidth = ceil((double)width / originalDilationFactor); //width of bigger subconvs
//        for(unsigned h = 0; h < height; ++h)
//        {
//            for(unsigned w = 0; w < width; ++w)
//            {
//                unsigned totalNumberOfRows=0;
//                unsigned totalNumberOfCols=0;

//                //calc number of rows
//                if((height % originalDilationFactor) == 0 || (h % originalDilationFactor)  < (height % originalDilationFactor))  // all the sub conv to the left are of full width
//                {
//                    totalNumberOfRows = h%originalDilationFactor * subConvHeight;
//                }
//                else
//                {
//                    //add height of subconvRows of full height first and then add remaining of smaller height
//                    totalNumberOfRows = (height % originalDilationFactor) * subConvHeight + (h%originalDilationFactor - height%originalDilationFactor)*(subConvHeight - 1);
//                }
//                totalNumberOfRows += h / originalDilationFactor;
//                //calc number of cols
//                if((width % originalDilationFactor) == 0 || (w % originalDilationFactor)  < (width % originalDilationFactor))  // all the sub conv to the left are of full width
//                {
//                    totalNumberOfCols = w%originalDilationFactor * subConvWidth;
//                }
//                else
//                {
//                    //add width*subConvWidth for of full subConvWidth  + (subConvWidth-1) for the rows of smaller subconvs
//                    totalNumberOfCols = (width % originalDilationFactor) * subConvWidth + (w%originalDilationFactor - width%originalDilationFactor)*(subConvWidth - 1);
//                }
//                totalNumberOfCols += w / originalDilationFactor;
//                unsigned subConvElementIdx = (totalNumberOfCols + totalNumberOfRows*width);

//                unsigned subConvElementOffset = subConvElementIdx * increment;

//                unpopulated_offsets[i++] = (subConvElementOffset << SHIFT_FOR_STORAGE_ELEMENT);
//            }
//        }
//    }

    activationStorageElement->populate(unpopulated_offsets, mv::Order("NHWC"));
}


//NOTE: The whole idea of the pwl is that we are going to use a linear function that represents leaky Relu.
//This comes through the equation and idea of Alessandro https://colab.research.google.com/drive/1xTQyJtZiPtMw-r1jUGks-aspbrpuEdKR#scrollTo=biQruEJ7olzD.
//Idea: We use the equation: ((x << m) + b) >> s, and train its variables in order to find a close solution that always satisfies the
//leaky relu. After we generate the instruction list table and we save the values of the registers inside.
//The map of the bits per instruction are described here: https://docs.google.com/spreadsheets/d/1RcD1FYGiKCTCRTDsU-J4r_FaQyAQbzMLyRu7WkeNoOY/edit#gid=0.
void populateInstructionListMap(mv::Data::TensorIterator instructionListTable)
{
    //NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    //27 of course will be aligned to 32 and will containt NOPS inside
    auto instructionListShape = instructionListTable->getShape();
    std::vector<uint32_t> template_table(instructionListShape.totalSize(), 0);

    //NOTE: first 2 are hardware reserved areas
    std::size_t ADDR_OF_RESERVED = 6;
    std::size_t ADDR_OF_ADDR_FLEX = 11;
    std::size_t ADDR_OF_FIRST2_BITS = 9;
    std::size_t ADDR_OF_REST_BITS = 16;
    std::size_t ADDR_OF_VALUE = 19;
    std::size_t MASK_FIRST2_BITS = 3;
    std::size_t first2_bits, last3_bits;
    std::vector<int> range_vector = {-128, -109, -90, -72, -54, -36, -18, 0, 128};
    std::vector<int> shift_vector = {1, -1, 0, 0, 0, -1, -1, -4};
    std::vector<int> bias_vector = {-119, 44, -43, -31, -19, 18, 10, 0};
    std::size_t k = 0;
    for (std::size_t j = 0; j < 32; j++)
    {
        first2_bits = j & MASK_FIRST2_BITS;
        last3_bits = j >> 2;

        if (j == 15)
            template_table[j] = (ALU_HALT_OPCODE);
        else if (j > 25)
            template_table[j] = (ALU_HALT_OPCODE);
        else
        {
            if (j < range_vector.size())
            {
                template_table[j] = ((range_vector[j] << ADDR_OF_VALUE) | (last3_bits << ADDR_OF_REST_BITS)
                    | (8 << ADDR_OF_ADDR_FLEX)
                    | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            }
            else if (j < range_vector.size() + shift_vector.size() + 1)
            {
                if (j < 16)
                    template_table[j] = ((shift_vector[j - range_vector.size()] << ADDR_OF_VALUE)
                        | (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX)
                        | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED)  | ALU_LOAD);
                 else
                {
                    k = j - 1;
                    first2_bits = k & MASK_FIRST2_BITS;
                    last3_bits = k >> 2;
                    template_table[j] = ((shift_vector[k - range_vector.size()] << ADDR_OF_VALUE)
                        | (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX)
                        | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED)  | ALU_LOAD);
                }
            }
            else if (j < range_vector.size() + shift_vector.size() + bias_vector.size() + 1)
            {
                k = j - 1;
                first2_bits = k & MASK_FIRST2_BITS;
                last3_bits = k >> 2;
                template_table[j] = ((bias_vector[k - range_vector.size() - shift_vector.size()] << ADDR_OF_VALUE)
                        | (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX)
                        | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED)  | ALU_LOAD);
            }
        }
    }

    std::vector<int64_t> template_table_appropriate_type(template_table.begin(), template_table.end());

    instructionListTable->populate(template_table_appropriate_type, mv::Order("NHWC"));
}

static void populateStorageElementPointersFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto op : om.getOps("DPUTask"))
    {
        auto taskOp = op->getOpType();
        if (taskOp == "DPUTask")
        {
            if(op->hasAttr("activationSparsityCompilerSolving") &&
                op->get<bool>("activationSparsityCompilerSolving"))
                populateActivationStorageElementMap(op, model);

            // New logic for generating SEP for dilated convolution
            if(op->hasAttr("activationSparsityCompilerSolvingForDilatedConv")
                    && op->get<bool>("activationSparsityCompilerSolvingForDilatedConv"))
            {
                populateActivationStorageElementMapForDilatedConvolution(op, model);
            }

            if(op->hasAttr("forcedToHaveActivationSparsityDueToDilatedConv")
                    && op->get<bool>("forcedToHaveActivationSparsityDueToDilatedConv"))
            {
                // NB this function still needs the correct logic to generate the SEPs
                populateActivationStorageElementMapForLayerAfterDilatedConvolution(op, model);
            }
        }
    }
}

static void populateInstructionListTablesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        auto taskOp = dpuTaskOp->getOpType();
        if (taskOp == "DPUTask")
        {
            if(dpuTaskOp->hasAttr("firstConvWithLRelu")
                    && dpuTaskOp->get<bool>("firstConvWithLRelu"))
            {
                auto instructionListTable
                        = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("instructionListTableIndex"));
                populateInstructionListMap(instructionListTable);
            }
        }
    }
}

static void generateWeightsTablesFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                std::string opName = dpuTaskOp->getName();
                std::string kernelWeightsTableName(mv::createWeightTableName(opName));

                auto outputChannels = dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                outputChannels = mv::round_up(dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                // per channel layout:
                // 3 -> bias
                // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
                // 1 -> SP_PTR
                // 0 -> DATA_PTR
                mv::Shape shape({WT_ELEMENTS_PER_CHANNEL, 1, 1, outputChannels});
                std::vector<int64_t> weightsTableData(shape.totalSize(), 0);
                mv::QuantizationParams quantParams = {{},{},{},{}};
                auto weightTable = om.constantInt(weightsTableData, shape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, kernelWeightsTableName);
                weightTable->set<bool>("weightTable", true);
                om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
                om.defineFlow(weightTable, dpuTaskOp, newSize - 1);
                dpuTaskOp->set<size_t>("weightsTableIndex", newSize - 1);
            }
        }
    }
}

static void generateInstructionListTablesFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOpType = dpuTaskOp->get<std::string>("taskOp");
            if((taskOpType == "Conv" || taskOpType == "ChannelMajorConvolution") && dpuTaskOp->hasAttr("postOpTypes") && dpuTaskOp->hasAttr("firstConvWithLRelu")
                    && dpuTaskOp->get<bool>("firstConvWithLRelu"))
            {
                auto ppeIterator = std::find(dpuTaskOp->get<std::vector<std::string>>("postOpTypes").begin(),
                                             dpuTaskOp->get<std::vector<std::string>>("postOpTypes").end(),
                                             "FLEXARB");
                if ( ppeIterator != dpuTaskOp->get<std::vector<std::string>>("postOpTypes").end())
                {
                    std::string opName = dpuTaskOp->getName();
                    std::string instructionListTableName(mv::createInstructionListTableName(opName));
                    std::size_t numberOfInstructions = 25;
                    std::size_t alignedInstructions = mv::round_up(numberOfInstructions, 16);
                    mv::Shape shape({alignedInstructions, 1, 1, 1});
                    std::vector<int64_t> instructionListTableData(shape.totalSize(), 0);
                    mv::QuantizationParams quantParams = {{},{},{},{}};
                    auto instructionListTable = om.constantInt(instructionListTableData, shape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, instructionListTableName);
                    instructionListTable->set<bool>("instructionListTable", true);
                    om.getSourceOp(instructionListTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                    unsigned newSize = dpuTaskOp->addInputTensor(instructionListTable);
                    om.defineFlow(instructionListTable, dpuTaskOp, newSize - 1);
                    dpuTaskOp->set<size_t>("instructionListTableIndex", newSize - 1);
                }
            }
        }
    }
}
