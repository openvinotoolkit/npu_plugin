#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include <math.h>

static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void populateSparseDataPointerMultiCluster(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, std::vector<int64_t> increments, long int offset, mv::ComputationModel& model);
static void populateDenseDataPointerMultiCluster(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, long int increment, long int offset, mv::ComputationModel& model);
static void removeBiasTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
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

    }
}

void populateSparseDataPointerMultiCluster(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, std::vector<int64_t> increments, long int offset, mv::ComputationModel &model)
{
    long int new_offset = offset;
    if (dpuTaskOp->get<std::string>("splitStrategy") != "SplitOverK")
    {
        for (size_t i = 0, k = 0; i < weightsTableData.size(); i+=4, )
        {
            // First increment is always 0
            offset += increments[k++];
            weightsTableData(i) = offset;
        }
    }
    else
    {
        auto globalParams = model.getGlobalConfigParams();
        unsigned numClusters = globalParams->get<int>("Number_of_Clusters");
        for (unsigned i = 0; i < numClusters; i++)
        {
            offset = new_offset;
            for (size_t j = 0, k = 0; j < weightsTableData.size()/numClusters; j+=4)
            {
                // First increment is always 0
                offset += increments[k++];
                weightsTableData(j + i * weightsTableData.size()/numClusters) = offset;
            }
        }
    }
    return;
}

void populateDenseDataPointerMultiCluster(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, long int increment, long int offset, mv::ComputationModel &model)
{
    long int new_offset = offset;
    if (dpuTaskOp->get<std::string>("splitStrategy") != "SplitOverK")
    {
        for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
              weightsTableData(i) = offset;
    }
    else
    {
        auto globalParams = model.getGlobalConfigParams();
        unsigned numClusters = globalParams->get<int>("Number_of_Clusters");
        for (unsigned i = 0; i < numClusters; i++)
        {
            offset = new_offset;
            for (size_t j = 0; j < weightsTableData.size()/numClusters; j+=4, offset +=increment)
            {
                weightsTableData(j + i * weightsTableData.size()/numClusters) = offset;
            }
        }
    }
    return;
}

void populateWeightsTablesDataPointers(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    if(dpuTaskOp->get<std::string>("taskOp") == "Conv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
        if(weights->isSparse())
        {
            auto tensorAllocatorName = weights->get<std::set<std::string>>("allocators").begin();
            auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
            mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, weights); // 0 is the only stage for now, but this will probably change in the future
            long int offset = tensorBufferIt->getOffset();
            std::vector<int64_t> increments = weights->getKernelDataOffsets();
            populateSparseDataPointerMultiCluster(weightsTableData, dpuTaskOp, increments, offset, model);
        }
        else
        {
            auto tensorAllocatorName = weights->get<std::set<std::string>>("allocators").begin();
            auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
            mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, weights); // 0 is the only stage for now, but this will probably change in the future
            long int offset = tensorBufferIt->getOffset();
            long int increment = weights->getShape()[0];
            populateDenseDataPointerMultiCluster(weightsTableData, dpuTaskOp, increment, offset, model);
        }
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" || dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
        auto tensorAllocatorName = weights->get<std::set<std::string>>("allocators").begin();
        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, weights); // 0 is the only stage for now, but this will probably change in the future
        long int offset = tensorBufferIt->getOffset();
        long int increment = weights->getShape()[0];
        populateDenseDataPointerMultiCluster(weightsTableData, dpuTaskOp, increment, offset, model);
    }
    // Max pooling does not need DataPointer

}

void populateWeightsTablesSparsityPointers(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto output = dpuTaskOp->getOutputTensor(0);
    unsigned outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

    if(dpuTaskOp->get<std::string>("taskOp") == "Conv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
        if(weights->isSparse())
        {
            auto weightsSparsityMap = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("sparsityMapIndex"));
            long int offset = weightsSparsityMap->getAddress();
            auto sparsityMapSizeInWords = weightsSparsityMap->getShape().totalSize();
            auto sparsityMapSizeInBytes = sparsityMapSizeInWords * weightsSparsityMap->getDType().getSizeInBits() / 8;
            auto sparsityMapBytesPerOutputChannel = sparsityMapSizeInBytes / outputChannels;
            long int increment = sparsityMapBytesPerOutputChannel;
            std::cout << weightsSparsityMap->getName() << " SM Bytes: " << std::to_string(sparsityMapSizeInBytes) << "/ output channels: " << outputChannels << " = Bytes p/OutputChannel: " << std::to_string(sparsityMapBytesPerOutputChannel) << std::endl;
            for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
            {
                weightsTableData(i+1) = offset;
                std::cout << "SparsityPointer Offset: " << std::to_string(offset) << std::endl;
            }
        }
        // Nothing to do here if is a dense ZMajor convolution
        else
        {
            long int offset = 16777215; // NOTE: Implementation defined
            for (size_t i = 0; i < weightsTableData.size(); i+=4)
                  weightsTableData(i+1) = offset;
        }
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" ||
            dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"  ||
            dpuTaskOp->get<std::string>("taskOp") == "MaxPool")
    {
        // We have fake sparsity here! Yuppi!
        auto activationWindow = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("fakeSparsityIndex"));
        auto activationWindowSizeInWords = activationWindow->getShape().totalSize();
        auto activationWindowSizeInBytes = activationWindowSizeInWords * activationWindow->getDType().getSizeInBits() / 8;
        auto activationWindowBytesPerOutputChannel = activationWindowSizeInBytes / outputChannels;
        auto tensorAllocatorName = activationWindow->get<std::set<std::string>>("allocators").begin();
        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, activationWindow); // 0 is the only stage for now, but this will probably change in the future
        long int offset = tensorBufferIt->getOffset();
        long int increment = activationWindowBytesPerOutputChannel;
        for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
              weightsTableData(i+1) = offset;
    }
}

void populateWeightsTablesActivationAndBias(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::DataModel dm(model);

    mv::QuantizationParams quantParams = {{},{},{},{}};
    auto output = dpuTaskOp->getOutputTensor(0);
    auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
    std::vector<int32_t> mScaled(outputChannels, 0);
    std::vector<int32_t> mShift(outputChannels, 0);
    if(output->hasAttr("quantParams"))
    {dpuTaskOp->get<unsigned>("outputChannels");
        quantParams = dpuTaskOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        if (!quantParams.isEmpty())
        {
            auto mult = quantParams.getMult();
            auto shift = quantParams.getShift();
            std::transform(mScaled.begin(), mScaled.end(), mult.begin(), mScaled.begin(), std::plus<int32_t>());
            std::transform(mShift.begin(), mShift.end(), shift.begin(), mShift.begin(), std::plus<int32_t>());
        }
    }
    std::vector<mv::DataElement> biasData;
    bool hasBias = dpuTaskOp->hasAttr("bias");
    mv::Data::TensorIterator bias;
    if (hasBias)
    {
        bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
        biasData = bias->getData(); //Bias has the type Int32 in both cases above
    }

    // per channel layout:
    // 3 -> bias
    // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
    // 1 -> SP_PTR
    // 0 -> DATA_PTR
    // TODO mult & prelu are currently not implemented

    unsigned round_mode = 1;
    std::vector<int32_t> round32(outputChannels, round_mode);

    for (size_t i = 0; i < weightsTableData.size(); i+=4)
    {
        weightsTableData(i+2) = static_cast<long int>((mScaled[i/4] << 16) | (round32[i/4] << 14) | (mShift[i/4]) << 8);

        if (hasBias)
            weightsTableData(i+3) = biasData[i/4];
    }
}

static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            if((dpuTaskOp->get<std::string>("taskOp") == "Conv") ||
               (dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTaskOp->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"))
            {
                // This pass is executed when there are not DMA Tasks yet, no hack needed
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("weightsTableIndex"));
                populateWeightsTablesActivationAndBias(*weightsTable, dpuTaskOp, model);
            }
        }
    }
}

static void removeBiasTensorsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::set<std::string> biasNamesToRemove;
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            if((dpuTaskOp->get<std::string>("taskOp") == "Conv") ||
               (dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTaskOp->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"))
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

static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            if((dpuTaskOp->get<std::string>("taskOp") == "Conv") ||
               (dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTaskOp->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"))
            {
                // Necessary hack since data is copied with DMA and we are not using a shared_ptr
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("weightsTableIndex"));
                auto weightsTableOp = om.getSourceOp(weightsTable);
                weightsTableOp = weightsTableOp.leftmostParent();
                weightsTable = weightsTableOp->getOutputTensor(0);

                populateWeightsTablesDataPointers(*weightsTable, dpuTaskOp, model);
                populateWeightsTablesSparsityPointers(*weightsTable, dpuTaskOp, model);
            }
        }
    }
}


static void generateWeightsTablesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            std::string taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                std::string opName = dpuTaskOp->getName();

                std::string kernelWeightsTableName(mv::createWeightTableName(opName));

                auto outputChannels = mv::round_up(dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                // per channel layout:
                // 3 -> bias
                // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
                // 1 -> SP_PTR
                // 0 -> DATA_PTR
                mv::Shape shape({4, 1, 1, outputChannels});

                std::vector<int64_t> weightsTableData(shape.totalSize(), 0);
                mv::QuantizationParams quantParams = {{},{},{},{}};

                auto weightTable = om.constantInt(weightsTableData, shape, mv::DType("Int32"), mv::Order("NWCH"), quantParams, kernelWeightsTableName);
                om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
                om.defineFlow(weightTable, dpuTaskOp, newSize - 1);
                dpuTaskOp->set<size_t>("weightsTableIndex", newSize - 1);
            }
        }
    }
}
