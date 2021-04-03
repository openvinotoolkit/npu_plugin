#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/op_model.hpp"

constexpr size_t MAX_CLUSTERS = 4;
constexpr size_t MAX_BASE_PTR = 512; // Register has 9 bits

static void assignSEBasePtrFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AssignSEBasePtr)
            .setFunc(assignSEBasePtrFcn)
            .setDescription(
                "Assigns base_ptrs to sparse tensors and marks segmented operations.");
    }
}

bool hasSegmentableStrategy(const mv::Data::OpListIterator& opIt)
{
    if (!opIt->hasAttr("splitStrategy"))
        return false;

    const std::vector<std::string> strategy = {"SplitOverH", "HKSwitch"};
    return std::find(strategy.cbegin(), strategy.cend(), opIt->get<std::string>("splitStrategy"))
           != strategy.cend();
}

void markSegmentedOps(mv::OpModel& om)
{
    auto isSegmented = [](const mv::Data::OpListIterator& opIt) {
        return opIt->getOpType() == "DPUTask" &&
               (opIt->get<std::string>("taskOp") == "Conv" || opIt->get<std::string>("taskOp") == "DepthwiseConv" || opIt->get<std::string>("taskOp") == "MaxPool") &&
               opIt->get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT] > 1 &&
               hasSegmentableStrategy(opIt);
    };

    const auto ops = om.getOps();
    for (auto& opIt : ops)
    {
        if (isSegmented(opIt))
            opIt->set<bool>("is_segmented", true);
    }
}

std::vector<mv::Data::OpListIterator> findConsumerOps(mv::DataModel& dm, const mv::Data::TensorIterator& tensor)
{
    std::vector<mv::Data::OpListIterator> consumerOps;
    const auto sinkLayers = mv::findSinkLayers(dm, tensor);
    for (auto& sink : sinkLayers)
    {
        if (sink->getOpType() == "DMATask" || sink->getOpType() == "Concat" || sink->isImplicit())
            for (auto& outputTensor : sink->getOutputTensor())
            {
                auto consumers = findConsumerOps(dm, outputTensor);
                consumerOps.insert(consumerOps.end(), consumers.begin(), consumers.end());
            }
        else
            consumerOps.push_back(sink);
    }
    return consumerOps;
}

std::set<unsigned short> getOpBaseIndices(const mv::Data::OpListIterator& opIt)
{
    std::set<unsigned short> basePtrs;

    for (auto inputTensor : opIt->getInputTensor())
    {
        if (inputTensor->hasAttr("base_ptrs"))
        {
            const auto producerBasePtrs = inputTensor->get<std::vector<unsigned short>>("base_ptrs");
            basePtrs.insert(producerBasePtrs.begin(), producerBasePtrs.end());
        }
    }

    return basePtrs;
}

std::vector<unsigned short> findAvailableBasePtrs(const std::set<unsigned short>& unavailableBasePtrs, const unsigned int numPtrs, const bool samePtr)
{
    std::vector<unsigned short> basePtrs(MAX_CLUSTERS, 0);
    unsigned int idx = 0;

    for (unsigned int i = 1; i < MAX_BASE_PTR; ++i)
        if (unavailableBasePtrs.find(i) == unavailableBasePtrs.end())
        {
            basePtrs[idx++] = i;
            if (samePtr || idx >= numPtrs)
                break;
        }

    if (samePtr && numPtrs > 1)
        for (; idx < numPtrs; ++idx)
            basePtrs[idx] = basePtrs[0];

    if (idx < numPtrs)
        throw std::runtime_error("Found only " + std::to_string(idx+1) + " base_ptrs available.");

    return basePtrs;
}

std::vector<unsigned short> findBasePtrs(mv::OpModel& om, mv::DataModel& dm, const mv::Data::TensorIterator& tensor, std::set<unsigned short>& unavailableBasePtrs, const unsigned int numClusters)
{
    auto originalTensor = tensor;
    auto producerOp = om.getSourceOp(tensor);
    while (producerOp->getOpType() == "DMATask" || producerOp->isImplicit())
    {
        originalTensor = producerOp->getInputTensor(0);
        producerOp = om.getSourceOp(originalTensor);
    }

    const auto consumerOps = findConsumerOps(dm, originalTensor);
    for (auto consumerOp : consumerOps)
    {
        const auto basePtrs = getOpBaseIndices(consumerOp);
        unavailableBasePtrs.insert(basePtrs.begin(), basePtrs.end());
    }

    const bool samePtr = tensor->hasAttr("splitStrategy") && tensor->get<std::string>("splitStrategy") != "SplitOverH";
    return findAvailableBasePtrs(unavailableBasePtrs, numClusters, samePtr);
}

void propagateInputBasePtrs(mv::OpModel& om, mv::Data::TensorIterator tensor, const std::vector<unsigned short>& basePtrs)
{
    tensor->set<std::vector<unsigned short>>("base_ptrs", basePtrs);

    auto producerOp = om.getSourceOp(tensor);
    if (producerOp->getOpType() == "DMATask" || producerOp->getOpType() == "Concat" || producerOp->isImplicit())
    {
        if (producerOp->getOpType() == "Concat" || producerOp->getOpType() == "ImplicitConcat")
        {
            for (auto& inputTensor : producerOp->getInputTensor())
                propagateInputBasePtrs(om, inputTensor, basePtrs);
        }
        else
        {
            producerOp = om.getSourceOp(tensor);
            propagateInputBasePtrs(om, producerOp->getInputTensor(0), basePtrs);
        }
    }
}

void propagateOutputBasePtrs(mv::DataModel& dm, mv::Data::TensorIterator tensor, const std::vector<unsigned short>& basePtrs)
{
    tensor->set<std::vector<unsigned short>>("base_ptrs", basePtrs);

    const auto consumerOps = mv::findSinkLayers(dm, tensor);
    for (auto& consumerOp : consumerOps)
        if (consumerOp->getOpType() == "DMATask" || consumerOp->getOpType() == "Concat" || consumerOp->isImplicit())
        {
            for (auto& outputTensor : consumerOp->getOutputTensor())
                propagateOutputBasePtrs(dm, outputTensor, basePtrs);
        }
}

void setInputBasePtrs(mv::OpModel& om, mv::DataModel& dm, const mv::Data::OpListIterator& opIt, const unsigned int numClusters)
{
    std::set<unsigned short> unavailableBasePtrs;
    if (opIt->get<std::string>("taskOp") == "Eltwise")
    {
        if (!opIt->getInputTensor(0)->hasAttr("base_ptrs"))
        {
            const auto basePtrs = findBasePtrs(om, dm, opIt->getInputTensor(0), unavailableBasePtrs, numClusters);
            propagateInputBasePtrs(om, opIt->getInputTensor(0), basePtrs);
            unavailableBasePtrs.insert(basePtrs.begin(), basePtrs.end());
        }
        if (!opIt->getInputTensor(1)->hasAttr("base_ptrs"))
        {
            const auto basePtrs = findBasePtrs(om, dm, opIt->getInputTensor(1), unavailableBasePtrs, numClusters);
            propagateInputBasePtrs(om, opIt->getInputTensor(1), basePtrs);
        }
    }
    else
    {
        if (!opIt->getInputTensor(0)->hasAttr("base_ptrs"))
        {
            const auto basePtrs = findBasePtrs(om, dm, opIt->getInputTensor(0), unavailableBasePtrs, numClusters);
            propagateInputBasePtrs(om, opIt->getInputTensor(0), basePtrs);
        }
    }
}

void setOutputBasePtrs(mv::OpModel& om, mv::DataModel& dm, const mv::Data::OpListIterator& opIt, const unsigned int numClusters)
{
    for (auto outputTensor : opIt->getOutputTensor())
    {
        std::set<unsigned short> unavailableBasePtrs;
        const auto basePtrs = findBasePtrs(om, dm, outputTensor, unavailableBasePtrs, numClusters);
        propagateOutputBasePtrs(dm, outputTensor, basePtrs);
    }
}

// Ticket VPUNND-3829 offers an overview on the motivation and the implementation
void assignSEBasePtrFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    markSegmentedOps(om);

    const auto numClusters = model.getGlobalConfigParams()->get<int>("Number_of_Clusters");

    const auto sortedOps = om.topologicalSort();
    for (auto& opIt : sortedOps)
    {
        if (opIt->getOpType() != "DPUTask")
            continue;

        if (opIt->hasAttr("inputActivationSparsity") && opIt->get<bool>("inputActivationSparsity"))
            setInputBasePtrs(om, dm, opIt, numClusters);

        if (opIt->hasAttr("outputActivationSparsity") && opIt->get<bool>("outputActivationSparsity"))
            setOutputBasePtrs(om, dm, opIt, numClusters);
    }
}
