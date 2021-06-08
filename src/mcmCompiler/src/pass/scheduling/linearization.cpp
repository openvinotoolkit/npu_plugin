#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"

#include <cassert>

using SizePair = std::pair<std::size_t, bool>;

struct OpInfo
{
    mv::Control::OpListIterator opIter;
    std::size_t totalSize;

    OpInfo(mv::Control::OpListIterator opIter_,
           std::size_t totalSize_) :
        opIter(opIter_), totalSize(totalSize_)  {}
};

// Forward declarations
static void linearizeFcn(const mv::pass::PassEntry& pass,
                         mv::ComputationModel& model,
                         mv::TargetDescriptor&,
                         mv::Element& passDesc,
                         mv::Element&);
namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(Linearization)
        .setFunc(linearizeFcn)
        .setDescription(
          "This pass linearizes the IR. It adds edges to the control flow to"
          " define deterministic order of exectution of unrelated operations."
        );
    }
}

// If the sizes are the same (should be very rare occurance),
// use the name of the operation to get a deterministic order.
bool operator<(const OpInfo& lhs, const OpInfo& rhs)
{
    if (lhs.totalSize == rhs.totalSize)
    {
        return
            (&*lhs.opIter)->getName().compare((&*rhs.opIter)->getName()) == -1;
    }

    return lhs.totalSize < rhs.totalSize;
}


// The DMA operations are sorted based on the sort of the DPU ones.
// We iterate trougs the DPU Ops in the sorted order,
// examine their input and outputs tensor and sort the DMA operations
// in that order. This places the DMA operations in the same order
// the DPU ones are executed.
static void sortDMAOps(std::vector<OpInfo>* dmaOpsPtr,
                                const std::vector<OpInfo>& dpuOps)
{
    if (dmaOpsPtr == nullptr || dpuOps.empty())
        return; // Nothing to sort.

    // Get the in and out tensors from the DPUs oredred list and order
    // the DMA list in such order, so the DMA operations happens 
    // in order the DPU ops use them.
    std::vector<OpInfo> orederedDmaOps = std::vector<OpInfo>();
    std::set<OpInfo*> addedDmaOps = std::set<OpInfo*>();
    for (auto& dpuOpInfo : dpuOps)
    {
        // Do the DPU input tensors.
        auto dpuInputTensors = (&*dpuOpInfo.opIter)->getInputTensor();
        for (auto& dpuInputTensor : dpuInputTensors)
        {
            for (auto& dmaOpInfo : *dmaOpsPtr)
            {
                if (addedDmaOps.find(&dmaOpInfo) != addedDmaOps.end())
                    continue;

                auto dmaOutputTensors =
                        (&*dmaOpInfo.opIter)->getOutputTensor();
                for (auto& dmaOutputTensor : dmaOutputTensors) 
                {
                    if (&*dmaOutputTensor == &*dpuInputTensor)
                    {
                        orederedDmaOps.emplace_back(dmaOpInfo);
                        addedDmaOps.insert(&dmaOpInfo);
                        break;
                    }
                }
            }
        }

        // Do the DPU output tensors (for the purposes of spilling, etc.)
        auto dpuOutputTensors = (&*dpuOpInfo.opIter)->getOutputTensor();
        for (auto& dpuOutputTensor : dpuOutputTensors)
        {
            for (auto& dmaOpInfo : *dmaOpsPtr)
            {
                if (addedDmaOps.find(&dmaOpInfo) != addedDmaOps.end())
                    continue;

                auto dmaInputTensors =
                        (&*dmaOpInfo.opIter)->getInputTensor();
                for (auto& dmaInputTensor : dmaInputTensors) 
                {
                    if (&*dmaInputTensor == &*dpuOutputTensor)
                    {
                        orederedDmaOps.emplace_back(dmaOpInfo);
                        addedDmaOps.insert(&dmaOpInfo);
                        break;
                    }
                }
            }
        }
    }

    // Now add the ops that were not added to the sorted list to the start.
    for (auto& dmaOpInfo : *dmaOpsPtr)
    {
        if (addedDmaOps.find(&dmaOpInfo) != addedDmaOps.end())
            continue;
        orederedDmaOps.emplace(orederedDmaOps.begin(), dmaOpInfo);
    }

    // Now place the ordered list in the original unordered list
    dmaOpsPtr->clear();

    for (auto& dmaOpInfo : orederedDmaOps)
    {
        dmaOpsPtr->emplace_back(dmaOpInfo);
    }
}

static void sortParallelLists(
      const mv::LogSender& logger,
      std::unordered_map<std::string,
            std::vector<OpInfo>>& sequentialLists)
{
    std::vector<OpInfo>* dmaOpsPtr = nullptr;
    std::vector<OpInfo>* dpuOps = nullptr;

    for (auto& listInfo : sequentialLists)
    {
        std::string listNameStart = listInfo.first.substr(0, 4);
        if (listNameStart != "DMA:")
        {
            std::stable_sort(listInfo.second.begin(), listInfo.second.end());
            if (listNameStart == "DPU:")
                dpuOps = &listInfo.second;
        }
        else
            dmaOpsPtr= &listInfo.second;
    }

    // Now sort the DMA ops based on the sorting of the DPU ops
    if (dpuOps != nullptr)
        sortDMAOps(dmaOpsPtr, *dpuOps);

    logger.log(mv::Logger::MessageType::Debug, "Sorted parallel lists");
}

static void insertEdges(const mv::LogSender& logger, mv::ControlModel& cm,
      std::unordered_map<std::string,
            std::vector<OpInfo>>& sequentialLists)
{
    for (auto& listInfo : sequentialLists)
    {
        std::size_t listSize = listInfo.second.size();
        if (listSize <= 1)
            // Nothing to do. Just exit.
            return;

        std::size_t index = 0;

        while(index < (listSize - 1))
        {
            mv::Control::OpListIterator firstOpIter = listInfo.second[index].opIter;
            mv::Control::OpListIterator secondOpIter = listInfo.second[index + 1].opIter;
            logger.log(mv::Logger::MessageType::Debug,
                    "Attempt to create edge between: " +
                    firstOpIter->getName() +
                    " and " + secondOpIter->getName());

            // If the flow is allowed and there is no dependency, add an edge.
            if (cm.isFlowAllowed(firstOpIter, secondOpIter) &&
                !cm.pathExists(firstOpIter, secondOpIter) &&
                !cm.pathExists(secondOpIter, firstOpIter))
            {
                mv::Control::FlowListIterator newEdge =
                    cm.defineFlow(firstOpIter, secondOpIter);
                if (newEdge == cm.flowEnd())
                {
                    logger.log(mv::Logger::MessageType::Debug,
                        "Couldn't create linearization control flow between: "
                        + firstOpIter->getName() +
                        " and " + secondOpIter->getName());
                }
                else
                {
                    logger.log(mv::Logger::MessageType::Debug,
                        "Added edge between ops: " +
                        firstOpIter->getName() + " and " +
                        secondOpIter->getName());
                }
            }

            index++;
        }

        logger.log(mv::Logger::MessageType::Debug,
                    "Edge additions done!");
    }
}

/// Calculates the size of the tensor t.
/// param: The tensor.
/// param: tensorSet The set of tensors
/// param: numClusters The number of clusters on the hardware.
/// return: a pair, the first member is the total size of the in and out
///         tensors, based on requested cluster. If the size is 0, the second
///         member is set to true. If the size can't be determined, the second
///         is set to false.
static  SizePair getTotalSize(const mv::LogSender& logger,
                                mv::Tensor* t,
                                std::set<mv::Tensor*>& tensorSet,
                                int numClusters)
{
    SizePair ret = {0, false};
    if (!t->hasSubTensors())
    {
        ret.first = t->computeTotalSize();
        ret.second = true;
    }
    else {
        for (int i = 0; i < numClusters; i++)
         {
            mv::Tensor& subTensor = t->getSubTensor(i);
            auto itSub = tensorSet.find(&subTensor);
            if (itSub != tensorSet.end())
            {
                mv::Tensor* tSub = *itSub;
                if (!tSub->hasAttr("allocators") ||
                        !t->get<std::set<std::string>>("allocators").
                            count("GraphFile"))
                    continue;
                ret.first += t->computeTotalSize();
                ret.second |= true;
            }
        }
    }

    logger.log(mv::Logger::MessageType::Debug,
        "Calculated total tensor size:" +
        std::to_string(ret.first) + ":" + std::to_string(ret.second));

    return ret;
}

/// Collect the input and ouput sizes for the tensors
/// associated with an operation.
/// param: op The operation
/// param: tensorSet The set of tensors
/// param: numClusters The number of clusters on the hardware.
/// return: a pair, the first member is the total size of the in and out
///         tensors, based on requested cluster. If the size is 0, the second
///         member is set to true. If the size can't be determined, the second
///         is set to false.
static SizePair getTotalSize(const mv::LogSender& logger,
                                mv::Op* op,
                                std::set<mv::Tensor*>& tensorSet,
                                int numClusters)
{
    SizePair ret = {0, false};

    for (auto& tensor : op->getInputTensor())
    {
        auto it = tensorSet.find(&*tensor);
        if (it != tensorSet.end())
        {
            auto tensorSize = getTotalSize(
                logger, *it, tensorSet, numClusters);
            ret.first += tensorSize.first;
            ret.second |= tensorSize.second;
        }
    }

    for (auto& tensor : op->getOutputTensor())
    {
        auto it = tensorSet.find(&*tensor);
        if (it != tensorSet.end())
        {
            auto tensorSize = getTotalSize(
                logger, *it, tensorSet, numClusters);
            ret.first += tensorSize.first;
            ret.second |= tensorSize.second;
        }
    }

    logger.log(mv::Logger::MessageType::Debug,
        "Linearization:Tensor's total size calculated for op:" +
        op->getName() + "-->" + std::to_string(ret.first) + ":" +
        std::to_string(ret.second));

    return ret;
}

static bool addOpInfoToList(
    const mv::LogSender& logger,
    const std::string& key,
    mv::Control::OpListIterator& opListIter,
    std::unordered_map<std::string,
    std::vector<OpInfo>>& sequentialLists,
    std::set<mv::Tensor*> tensorSet,
    int numClusters)
{
    auto szPair = getTotalSize(
        logger, &*opListIter, tensorSet, numClusters);
    if (szPair.second == 0 && szPair.second)
    {
        // If the size of the tensor (for the specified cluster is 0 and
        // it is being explicitly set - not unknown), don't add this node
        // the list of OpInfos to sort and add edge to.
        logger.log(mv::Logger::MessageType::Debug,
            "Not added node for creating an edge (size=0): " +
                (&*opListIter)->getName());
        return false;
    }

    OpInfo opInfo = OpInfo{opListIter, szPair.first};

    // If element exists, no insertion happens. It returns the
    // iterator to the existing element.
    auto iter = sequentialLists.emplace(key,
                    std::vector<OpInfo>());
    iter.first->second.push_back(opInfo);

    logger.log(mv::Logger::MessageType::Debug,
        "Added node for creating an edge: " +
            (&*opInfo.opIter)->getName() + ":" +
            std::to_string(opInfo.totalSize));
    return true;
}

void linearizeFcn(const mv::pass::PassEntry& pass,
                  mv::ComputationModel& model,
                  mv::TargetDescriptor&,
                  [[maybe_unused]]mv::Element& passDesc,
                  mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    auto globalConfig = model.getGlobalConfigParams();

    // If the inclusion of this pass is not explicit in the config file,
    // Don't perform the linearization.
    if (!globalConfig->hasAttr("enable_operation_serialization") ||
        !globalConfig->get<bool>("enable_operation_serialization"))
    {
        return;
    }

    mv::DataModel dataModel(model);
    int numClusters =
             dataModel.getGlobalConfigParams()->get<int>(
                                            "Number_of_Clusters");

    mv::ControlModel controlModel(model);

    // Populate the map with lists for the divices.
    std::unordered_map<std::string,
        std::vector<OpInfo>> sequentialLists =
                        std::unordered_map<std::string,
                            std::vector<OpInfo>>();
    // 1. Clasify the operation into different
    // "parralel (for execution)" lists.
    std::set<mv::Tensor*> tensorSet;
    for (auto t = model.tensorBegin(); t != model.tensorEnd(); ++t)
    {
        if (!t->hasAttr("allocators") ||
            !t->get<std::set<std::string>>("allocators").count("GraphFile"))
                continue;

        tensorSet.emplace(&*t);
    }

    for(auto op = controlModel.opBegin(); op != controlModel.opEnd(); ++op) {
        if (op->getOpType() == "DMATask")
        {
            std::string key = "DMA:";
            addOpInfoToList(
                pass, key, op, sequentialLists, tensorSet, numClusters);

            pass.log(mv::Logger::MessageType::Debug,
                "Added a DMATask op: key: " + key);
        }
        else if (op->getOpType() == "DPUTask")
        {
            // If element exists, no insertion happens. It returns the
            // iterator to the existing element.
            if (addOpInfoToList(
                pass, "DPU:0", op, sequentialLists, tensorSet, numClusters)) {
                pass.log(mv::Logger::MessageType::Debug,
                    "Added a DPUTask op for cluster 0."
                    " Subclusters not found.");
            }
            else
            {
                pass.log(mv::Logger::MessageType::Debug,
                    "Not Added a DPUTask op for cluster 0."
                    " subclusters not found.");
            }
        }
        else if (op->getOpType() == "UPATask")
        {
            std::string key = "UPA:";
            addOpInfoToList(
                pass, key, op, sequentialLists, tensorSet, numClusters);
            pass.log(mv::Logger::MessageType::Debug,
                   "Added a UPATask op");
        }
    }

    // 2. Sort the list, so we get deterministic order.
    sortParallelLists(pass, sequentialLists);

    // 3. Create edges between the sequesntial operations, so they are
    //    executed in deterministic order.
    insertEdges(pass, controlModel, sequentialLists);

    pass.log(mv::Logger::MessageType::Debug,
                   "Done with linearization pass.");
}
