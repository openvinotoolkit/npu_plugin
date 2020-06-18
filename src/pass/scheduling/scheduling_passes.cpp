#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/target/kmb/workloads.hpp"
#include <cassert>
#include <cmath>
#include <numeric>
#include <cmath>
#include <unordered_set>

using std::to_string;

static void generateSchedulingFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void barrierIndexAssignmentFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateBarrierRefsFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void storeBarriersNamesInTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateCountsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void hackExecutionScheduleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&);
static void correctExecutionScheduleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&);
static void reorderDmasInScheduleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&);
static void layoutDMAFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(CorrectExecutionSchedule)
        .setFunc(correctExecutionScheduleFcn)
        .setDescription(
            "Corrects a schedule for the computational model"
        );

        MV_REGISTER_PASS(GenerateExecutionSchedule)
        .setFunc(generateSchedulingFcn)
        .setDescription(
            "Generates a schedule for the computational model"
        );

        MV_REGISTER_PASS(BarrierIndexAssignment)
        .setFunc(barrierIndexAssignmentFcn)
        .setDescription(
            "Assigns a dynamic index to the barrier if a schedule has been produced"
        );

        MV_REGISTER_PASS(UpdateBarrierRefs)
        .setFunc(updateBarrierRefsFcn)
        .setDescription(
            "Updates the barrier refs on tasks using indices produced by BarrierIndexAssignment pass"
        );

        MV_REGISTER_PASS(StoreBarriersNamesInTasks)
        .setFunc(storeBarriersNamesInTasksFcn)
        .setDescription(
            "For each task barrier produced and consumed are stored as strings"
        );

        MV_REGISTER_PASS(UpdateBarrierProducerConsumerCounts)
        .setFunc(updateCountsFcn)
        .setDescription(
            "This pass updates producer and consumer counts in barriers based on workloads in producer and consumer \
            DxxTasks in the compute graph"
        );

        MV_REGISTER_PASS(HackExecutionSchedule)
        .setFunc(hackExecutionScheduleFcn)
        .setDescription(
            "This pass is intended for debug use only"
        );

        MV_REGISTER_PASS(ReorderDmasInSchedule)
        .setFunc(reorderDmasInScheduleFcn)
        .setDescription(
            "This pass reorders DMAs emanating from a given barrier task so that a DMA task consumed earlier will \
            get a lower scheduling number"
        );

        MV_REGISTER_PASS(LayoutDMA)
        .setFunc(layoutDMAFcn)
        .setDescription(
            "This pass optimizes DMA targets and port assignments"
        );

    }
}

// ASSUMPTION: DMA for weights, weights table, sparsity map and input are swappable in terms of scheduling without causing the model to hang on barriers
// This basically means that they have the same barrier dependencies
// ASSUMPTION 2: The correct dma order is: Weights - Sparsity Map (if present) - Weights Table - Input data (if present)
void correctExecutionScheduleFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passArg, mv::Element &)
{
    mv::ControlModel cm(model);
    mv::OpModel om(model);
    auto ops = cm.schedulingSort();

    bool inputFlag = true;

    if(passArg.hasAttr("inputFlag"))
        inputFlag = passArg.get<bool>("inputFlag");

    for(auto& op: ops)
    {
        std::string opType = op->getOpType();
        if(opType == "DPUTask")
        {
            std::string taskOp = op->get<std::string>("taskOp");
            if(taskOp == "Conv")
            {
                std::vector<unsigned> schedulingNumbers;

                if(inputFlag)
                {
                    auto inputTensor = op->getInputTensor(0);
                    auto inputTensorOp = om.getSourceOp(inputTensor);
                    if(inputTensorOp->getOpType() == "DMATask")
                        schedulingNumbers.push_back(inputTensorOp->get<unsigned>("schedulingNumber"));

                }

                auto weightsTensor = op->getInputTensor(1);
                auto weightsOp = om.getSourceOp(weightsTensor);
                schedulingNumbers.push_back(weightsOp->get<unsigned>("schedulingNumber"));
                auto weightsTableOp = om.getSourceOp(op->getInputTensor(op->get<std::size_t>("weightsTableIndex")));
                schedulingNumbers.push_back(weightsTableOp->get<unsigned>("schedulingNumber"));

                if(weightsTensor->isSparse())
                {
                    auto weightsSparsityMapOp = om.getSourceOp(op->getInputTensor(op->get<std::size_t>("sparsityMapIndex")));
                    schedulingNumbers.push_back(weightsSparsityMapOp->get<unsigned>("schedulingNumber"));
                }

                std::sort(schedulingNumbers.begin(), schedulingNumbers.end());
                unsigned currentIndex = 0;
                weightsOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);
                if(weightsTensor->isSparse())
                {
                    auto weightsSparsityMapOp = om.getSourceOp(op->getInputTensor(op->get<std::size_t>("sparsityMapIndex")));
                    weightsSparsityMapOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);
                }
                weightsTableOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);
                if(inputFlag)
                {
                    auto inputTensor = op->getInputTensor(0);
                    auto inputTensorOp = om.getSourceOp(inputTensor);
                    if(inputTensorOp->getOpType() == "DMATask")
                        inputTensorOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);
                }
            }
            else if(taskOp == "DepthwiseConv")
            {
                std::vector<unsigned> schedulingNumbers;

                if(inputFlag)
                {
                    auto inputTensor = op->getInputTensor(0);
                    auto inputTensorOp = om.getSourceOp(inputTensor);
                    if(inputTensorOp->getOpType() == "DMATask")
                        schedulingNumbers.push_back(inputTensorOp->get<unsigned>("schedulingNumber"));

                }

                auto weightsTensor = op->getInputTensor(1);
                auto weightsOp = om.getSourceOp(weightsTensor);
                schedulingNumbers.push_back(weightsOp->get<unsigned>("schedulingNumber"));
                auto weightsTableOp = om.getSourceOp(op->getInputTensor(op->get<std::size_t>("weightsTableIndex")));
                schedulingNumbers.push_back(weightsTableOp->get<unsigned>("schedulingNumber"));

                auto weightsSparsityMapOp = om.getSourceOp(op->getInputTensor(op->get<std::size_t>("fakeSparsityIndex")));
                schedulingNumbers.push_back(weightsSparsityMapOp->get<unsigned>("schedulingNumber"));

                std::sort(schedulingNumbers.begin(), schedulingNumbers.end());
                unsigned currentIndex = 0;
                weightsOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);

                if(inputFlag)
                {
                    auto inputTensor = op->getInputTensor(0);
                    auto inputTensorOp = om.getSourceOp(inputTensor);
                    if(inputTensorOp->getOpType() == "DMATask")
                        inputTensorOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);
                }

                weightsSparsityMapOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);
                weightsTableOp->set<unsigned>("schedulingNumber", schedulingNumbers[currentIndex++]);

            }
        }
    }

}

void hackExecutionScheduleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element &)
{

    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("schedule_helper_indices"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No schedule helper indices provided");
        return;
    }

    auto addressList = globalParams->get<std::vector<mv::Element>>("schedule_helper_indices");
    for (auto e : addressList)
    {
        std::string& name = e.get<std::string>("name_filter");
        int64_t address = e.get<int>("index");
        try
        {
            auto opIt = model.getOp(name);
            opIt->set<unsigned>("schedulingNumber", address);
        }
        catch (mv::ArgumentError error)
        {
            pass.log(mv::Logger::MessageType::Error, error.what());
        }

    }
}

std::vector<std::string> getBarriersProduced(mv::Data::OpListIterator task)
{
    if(task->hasAttr("BarriersProducedByTask"))
        return task->get<std::vector<std::string>>("BarriersProducedByTask");
    else
        return std::vector<std::string>();
}

std::vector<std::string> getBarriersConsumed(mv::Data::OpListIterator task)
{
    if(task->hasAttr("BarriersConsumedByTask"))
        return task->get<std::vector<std::string>>("BarriersConsumedByTask");
    else
        return std::vector<std::string>();
}

std::vector<std::string> getBarriersNeeded(mv::Data::OpListIterator task)
{
    std::vector<std::string> barriersNeeded;

    auto barriersProducedByTask = getBarriersProduced(task);
    for(auto& barrier : barriersProducedByTask)
        barriersNeeded.push_back(barrier);

    auto barriersConsumedByTask = getBarriersConsumed(task);
    for(auto& barrier : barriersConsumedByTask)
        barriersNeeded.push_back(barrier);

    return barriersNeeded;
}

bool isInsertable(mv::ComputationModel& model, const std::string& taskName, const std::unordered_set<std::string>& barriersInMemory)
{
    // A task is insertable if:
    // 1) The barriers it needs are pushable in barriersInMemory without overflowing the 8 barriers in memory limit
    // 2) The barriers it needs are already in barriersInMemory

    auto task = model.getOp(taskName);

    std::vector<std::string> barriersNeeded = getBarriersNeeded(task);

    unsigned barriersInMemorySize = barriersInMemory.size();
    unsigned barriersToInsert = 0;
    for(auto& barrier: barriersNeeded)
        if(!barriersInMemory.count(barrier))
            ++barriersToInsert;

    if(barriersInMemorySize + barriersToInsert > 8)
        return false;
    else
        return true;
}

std::vector<std::string> updateBarriersInMemoryForInsertion(mv::ComputationModel& model, const std::string& taskName, std::unordered_set<std::string>& barriersInMemory, std::unordered_set<std::string>& availableTasks)
{
    // When task is pushed to the queue the following things happen
    // 0) If the barriers involved (produced and consumed) with the task are not in memory, push them to both barriersInMemory and toReturn vector.
    // 1) Find the barriers in memory produced by the task, reduce the number of producers by 1. If any of these barriers reaches zero producers, push its consumers to availableTasks
    // 2) Find the barriers in memory consumed by the task, reduce the number of consumers by 1. If any of these barriers reaches zero consumers, remove it from the barriersInMemory list
    auto task = model.getOp(taskName);
    std::vector<std::string> toReturn;

    std::vector<std::string> barriersNeeded = getBarriersNeeded(task);
    for(auto& barrier : barriersNeeded)
    {
        if(!barriersInMemory.count(barrier))
        {
            toReturn.push_back(barrier);
            barriersInMemory.insert(barrier);
        }
    }

    auto barriersProduced = getBarriersProduced(task);
    for(auto& barrier : barriersProduced)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        physicalBarrier.setNumProducers(physicalBarrier.getNumProducers() - 1);
        if(physicalBarrier.getNumProducers() == 0)
            for(auto& consumer : physicalBarrier.getConsumers())
                availableTasks.insert(consumer);

    }

    auto barriersConsumed = getBarriersConsumed(task);
    for(auto& barrier : barriersConsumed)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        physicalBarrier.setNumConsumers(physicalBarrier.getNumConsumers() - 1);
        if(physicalBarrier.getNumConsumers() == 0)
            barriersInMemory.erase(barrier);

    }

    return toReturn;
}

void updateBarriersInMemoryForRemotion(mv::ComputationModel& model, const std::string& taskName, std::unordered_set<std::string>& barriersInMemory, std::unordered_set<std::string>& availableTasks, std::vector<std::string>& addedBarriers)
{
    // When task is removed from the queue the following things happen
    // 1) Find the barriers produced by the task, increment the number of producers by 1. If the previous number was 0, its consumers must be removed from the list of available tasks
    // 2) Find the barriers in memory consumed by the task, increment the number of consumers by 1. If the previous number was 0, add the barriers to barriersInMemory list
    // 3) For each barrier in addedBarriers, remove it from barriersInMemory

    auto task = model.getOp(taskName);
    auto barriersProduced = getBarriersProduced(task);
    for(auto& barrier : barriersProduced)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        unsigned numProducers = physicalBarrier.getNumProducers();
        physicalBarrier.setNumProducers(numProducers + 1);
        if(numProducers == 0)
            for(auto& consumer : physicalBarrier.getConsumers())
                availableTasks.erase(consumer);

    }

    auto barriersConsumed = getBarriersConsumed(task);
    for(auto& barrier : barriersConsumed)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        unsigned numConsumers = physicalBarrier.getNumConsumers();
        physicalBarrier.setNumConsumers(numConsumers + 1);
        if(numConsumers == 0)
            barriersInMemory.insert(barrier);
    }

    for(auto& barrier : addedBarriers)
        barriersInMemory.erase(barrier);

}



bool generateSchedulingRecursively(mv::ComputationModel& model, std::unordered_set<std::string>& availableTasks, std::vector<std::string>& scheduling, std::unordered_set<std::string>& barriersInMemory)
{
    std::vector<std::string> pushableTasks;

    for(auto& task: availableTasks)
        if(isInsertable(model, task, barriersInMemory))
            pushableTasks.push_back(task);

    for(auto& pushableTask: pushableTasks)
    {
        // Push task into scheduling (and thus removing it from available tasks list)
        scheduling.push_back(pushableTask);
        availableTasks.erase(pushableTask);

        //Update the barriers in memory due to newly inserted task
        auto addedBarriers = updateBarriersInMemoryForInsertion(model, pushableTask, barriersInMemory, availableTasks);

        if(barriersInMemory.empty())
            return true;

        if(generateSchedulingRecursively(model, availableTasks, scheduling, barriersInMemory))
            return true;
        else
        {
            //Can't build a schedule, we have to reset the memory structures as they were
            //before trying the next operation
            scheduling.erase(scheduling.end() - 1);
            availableTasks.insert(pushableTask);
            updateBarriersInMemoryForRemotion(model, pushableTask, barriersInMemory, availableTasks, addedBarriers);
        }
    }

    // If we arrived here, it means we tried all pushable tasks
    // It's impossible to produce a schedule at this point as we can't push nothing
    return false;
}

void generateSchedulingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::ControlModel cm(model);

    auto barrierTasks = model.getOps("BarrierTask");
    unsigned numTasks = 0;

    std::unordered_set<std::string> availableTasks;
    for(auto opIt = cm.opBegin(); opIt != cm.opEnd(); ++opIt)
    {
        if(opIt->getOpType().find("Task") != std::string::npos && opIt->getOpType() != "BarrierTask")
        {
            availableTasks.insert(opIt->getName());
            ++numTasks;
        }
    }

    // Find the tasks that have no barrier dependency, we have to do it through barriers
    // by successive eliminations
    for(auto& barrierTask : barrierTasks)
    {
        auto barrier = barrierTask->get<mv::Barrier>("Barrier");
        auto consumersNames = barrier.getConsumers();
        for(auto& consumerName: consumersNames)
        {
            if(availableTasks.count(consumerName))
                availableTasks.erase(consumerName);
        }
    }

    // Remove trailing UPATasks from availableTasks
    auto upaTasks = cm.getOps("UPATask");
    for (auto& task : upaTasks)
        if (task->hasAttr("trailing") && task->get<bool>("trailing"))
            availableTasks.erase(task->getName());

    std::vector<std::string> scheduling;
    std::unordered_set<std::string> barriersInMemory;
    if(!availableTasks.empty() && !generateSchedulingRecursively(model, availableTasks, scheduling, barriersInMemory))
        throw "Impossible to schedule";

    unsigned i = 0;
    for(auto& task : scheduling)
        model.getOp(task)->set<unsigned>("schedulingNumber", i++);

    // Schedule trailing UPATasks
    mv::OpModel om(model);
    for (auto task = om.opBegin(); task != om.opEnd(); ++task)
    {
        if (task->hasAttr("trailing") && task->get<bool>("trailing"))
            if (std::find(scheduling.begin(), scheduling.end(), task->getName()) == scheduling.end())
                task->set<unsigned>("schedulingNumber", i++);
    }
}

void barrierIndexAssignmentFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::ControlModel cm(model);
    mv::OpModel om(model);

    auto globalConfigParams = model.getGlobalConfigParams();
    std::string indexAssignment = globalConfigParams->get<std::string>("barrier_index_assignment");

    if (indexAssignment == "Dynamic")
    {
        auto sortedOps = cm.schedulingSort();

        int id = 0;
        for (auto op: sortedOps)
        {
            auto barriers = getBarriersNeeded(om.switchContext(op));
            for(auto& barrier : barriers)
            {
                auto barrierTask = model.getOp(barrier);
                auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
                //If the index was not set before
                if(physicalBarrier.getIndex() == -1)
                    physicalBarrier.setIndex(id++);
            }
        }
    }
}

void storeBarriersNamesInTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto barrierTasks = om.getOps("BarrierTask");

    for (auto bt: barrierTasks)
    {
        auto& barrier = bt->get<mv::Barrier>("Barrier");

        for (auto producer: barrier.getProducers())
        {
            auto producerOp = om.getOp(producer);

            if (!producerOp->hasAttr("BarriersProducedByTask"))
                producerOp->set<std::vector<std::string>>("BarriersProducedByTask", std::vector<std::string>());

            auto& barrierRef = producerOp->get<std::vector<std::string>>("BarriersProducedByTask");
            barrierRef.push_back(bt->getName());
        }

        for (auto consumer: barrier.getConsumers())
        {
            auto consumerOp = om.getOp(consumer);
            if (!consumerOp->hasAttr("BarriersConsumedByTask"))
                consumerOp->set<std::vector<std::string>>("BarriersConsumedByTask", std::vector<std::string>());

            auto& barrierRef = consumerOp->get<std::vector<std::string>>("BarriersConsumedByTask");
            barrierRef.push_back(bt->getName());
        }
    }
}

void updateBarrierRefsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto barrierTasks = om.getOps("BarrierTask");

    for (auto bt: barrierTasks)
    {
        auto& barrier = bt->get<mv::Barrier>("Barrier");

        for (auto producer: barrier.getProducers())
        {
            auto producerOp = om.getOp(producer);

            if (!producerOp->hasAttr("BarrierDeps"))
                producerOp->set<mv::BarrierDependencies>("BarrierDeps", mv::BarrierDependencies());

            auto& barrierRef = producerOp->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.addUpdateBarrier(barrier.getIndex());
        }

        for (auto consumer: barrier.getConsumers())
        {
            auto consumerOp = om.getOp(consumer);
            if (!consumerOp->hasAttr("BarrierDeps"))
                consumerOp->set<mv::BarrierDependencies>("BarrierDeps", mv::BarrierDependencies());

            auto& barrierRef = consumerOp->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.addWaitBarrier(barrier.getIndex());
        }
    }
}

// DEPRECATED
// NEEDS TO BE UPDATED WHEN MERGING WITH MULTICLUSTER
void updateCountsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto barrierTasks = om.getOps("BarrierTask");

    for (auto bt: barrierTasks)
    {
        auto& barrier = bt->get<mv::Barrier>("Barrier");

        for (auto producer: barrier.getProducers())
        {
            auto producerOp = om.getOp(producer);
            if (producerOp->hasAttr("Workloads"))
            {
                auto& workloads = producerOp->get<mv::Workloads>("Workloads");
                int count = barrier.getNumProducers();
                count += workloads.nWorkloads();
                barrier.setNumProducers(count);
            }
            else
            {
                int count = barrier.getNumProducers();
                count += 1;
                barrier.setNumProducers(count);
            }
        }

        for (auto consumer: barrier.getConsumers())
        {
            auto consumerOp = om.getOp(consumer);

            if (consumerOp->hasAttr("Workloads"))
            {
                auto& workloads = consumerOp->get<mv::Workloads>("Workloads");
                int count = barrier.getNumConsumers();
                count += workloads.nWorkloads();
                barrier.setNumConsumers(count);
            }
            else
            {
                int count = barrier.getNumConsumers();
                count += 1;
                barrier.setNumConsumers(count);
            }
        }
    }
}

void reorderDmasInScheduleFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::ControlModel cm(model);

    std::vector<mv::Control::OpListIterator> dmaSrcTasks;
    for (auto op = cm.opBegin(); op != cm.opEnd(); ++op)
    {
        if (op->getOpType() == "Input" || op->getOpType() == "BarrierTask")
            dmaSrcTasks.push_back(op);
    }

    std::sort(dmaSrcTasks.begin(), dmaSrcTasks.end(), [](mv::Control::OpListIterator b1, mv::Control::OpListIterator b2)
    {
        return b1->get<unsigned>("layerNumber") < b2->get<unsigned>("layerNumber");
    });

    for (auto task : dmaSrcTasks)
    {
        // create an ordered map frm barrier layerNumber -> associated DMA.
        std::map<unsigned, std::vector<mv::Control::OpListIterator>> downstreamBarriers;
        unsigned minSchedulingNumber = INT_MAX;
        for (auto child = task.leftmostChild(); child != cm.opEnd(); ++child)
        {
            if (child->getOpType() == "DMATask" && child->get<mv::DmaDirection>("direction") == mv::DDR2NNCMX)
            {
                auto schedNum = child->get<unsigned>("schedulingNumber");
                if (schedNum < minSchedulingNumber)
                    minSchedulingNumber = schedNum;
                for (auto outgoingOp = child.leftmostChild(); outgoingOp != cm.opEnd(); ++outgoingOp)
                {
                    if (outgoingOp->getOpType() == "BarrierTask")
                    {
                        downstreamBarriers[outgoingOp->get<unsigned>("layerNumber")].push_back(child);
                    }
                }
            }
        }

        if (downstreamBarriers.empty())
            continue;

        for (auto it = downstreamBarriers.begin(); it != downstreamBarriers.end(); ++it)
        {
            auto dmaOps = it->second;

            for (auto dma: dmaOps)
            {
                dma->set<unsigned>("schedulingNumber", minSchedulingNumber);
                minSchedulingNumber++;
            }
        }

    }
}

struct OpInfo {
    OpInfo(mv::Op* op_, bool isDMA_, std::uint64_t latencyNS_) : op{op_}, latencyNS{latencyNS_}, csramNS{latencyNS_}, isDMA{isDMA_} {}
    OpInfo(mv::Op* op_, bool isDMA_, std::uint64_t latencyNS_, std::uint64_t csramNS_, std::uint64_t size_) : op{op_}, latencyNS{latencyNS_}, csramNS{csramNS_}, size{size_}, isDMA{isDMA_} {}

    std::uint64_t completeNS() const { return startNS + latencyNS; }
    std::uint64_t completeNSUsingCSRAM() const { return startNS + csramNS; }

    mv::Op* op;
    std::uint64_t latencyNS;
    std::uint64_t csramNS;
    std::uint64_t size = 0;
    std::uint64_t startNS = 0;
    bool isDMA;
    std::size_t portIdx = 0;
    mv::BarrierDependencies deps;
};

struct BarrierInfo
{
    std::uint64_t startNS;
    std::uint64_t earliestNS;
    std::vector<OpInfo*> waitOpInfos;
    std::vector<OpInfo*> updateOpInfos;
    std::vector<std::uint64_t> portSlackNS;
};

struct PortInfo
{
    std::uint64_t busyUntil;
};

OpInfo analyzeOp(const mv::LogSender& logger, mv::Op& op)
{
    logger.log(mv::Logger::MessageType::Debug,
               "Modeling task:\n" + op.toString() + "\n");

    if (op.getOpType() == "DPUTask")
    {
        // TODO: Better models for DPU tasks.
        //
        // For now, we assume that DPU tasks execute in time proportional
        // to their total IO.
        //
        // What we want is something like:
        // cycles = product of (output_dimension / mpe_mode) * kernel width * kernel height,
        // if conv: cycles *= (input dim 2 / channel conv mac config)
        // total = cycles * sparsity / efficiency / frequency

        std::uint64_t ioSize = 0;
        for (auto& tensor : op.getInputTensor())
        {
            ioSize += tensor->size();
        }
        for (auto& tensor : op.getOutputTensor())
        {
            ioSize += tensor->size();
        }
        ioSize /= 100;  // Just a guess -- 10x DDR speed
        return OpInfo{&op, false, ioSize};
    }

    if (op.getOpType() == "DMATask")
    {
        // We assume that DMA tasks execute in time proportional to
        // the smaller of the input tensor size and output tensor size
        // (which might not be equal, due to slicing), with a
        // multiplier dependent on the DMA direction.
        std::uint64_t inSize = 0;
        for (auto& tensor : op.getInputTensor())
        {
            logger.log(mv::Logger::MessageType::Debug,
                       "Input Tensor:\n" + tensor->toString()
                       + "\n  size=" + to_string(tensor->computeTotalSize())
                       + "\n  shape=" + tensor->getShape().toString()
                       + "\n");
            inSize += tensor->computeTotalSize();
        }

        std::uint64_t outSize = 0;
        for (auto& tensor : op.getOutputTensor())
        {
            logger.log(mv::Logger::MessageType::Debug,
                       "Output Tensor:\n" + tensor->toString()
                       + "\n  size=" + to_string(tensor->computeTotalSize())
                       + "\n  shape=" + tensor->getShape().toString()
                       + "\n");
            outSize += tensor->computeTotalSize();
        }

        std::uint64_t ioSize = std::min(inSize, outSize);

        // Convert to nanos by dividing by the gbps rate of the transfer.
        const unsigned ddrBw = 20;
        const unsigned cmxBw = 32;

        switch (op.get<mv::DmaDirection>("direction"))
        {
            case mv::DmaDirectionEnum::DDR2NNCMX:
                return OpInfo{&op, true, ioSize / ddrBw, ioSize / cmxBw, ioSize};
            case mv::DmaDirectionEnum::NNCMX2DDR:
                ioSize /= 20;
                return OpInfo{&op, true, ioSize / ddrBw};
            case mv::DmaDirectionEnum::CSRAM2NNCMX:
                return OpInfo{&op, true, ioSize / cmxBw};
            default:
                return OpInfo{&op, true, 0};  // Don't account for this DMA
        }
    }

    return OpInfo{&op, false, 0};  // TODO: What other tasks should we handle here?
}

// Simulates running the model, returning the resulting overall latency.
//
// As a side effect, this function recomputes:
// ) The DMA port assignments for each operation
// ) The time at which each barrier is expected to become ready
// ) The time when each operation is expected to start
std::uint64_t simRunModel(const mv::LogSender& logger,
                          std::vector<OpInfo>* opInfos,
                          std::vector<BarrierInfo>* barrierInfos,
                          std::vector<PortInfo>* portInfos)
{
    for (auto& portInfo : *portInfos)
    {
        portInfo = PortInfo{0};
    }

    for (auto& barrierInfo : *barrierInfos)
    {
        barrierInfo.startNS = 0;
        barrierInfo.earliestNS = 0;
    }

    const auto portLimit = portInfos->size();

    // Compute per-barrier latencies, including port effects.
    std::uint64_t overallLatency = 0;
    for (auto& opInfo : *opInfos)
    {
        std::string logMsg = "Scanning Op=" + opInfo.op->getName()
          + " ty=" + opInfo.op->getOpType()
          + " size=" + to_string(opInfo.size)
          + " wait=";
        for (auto wait : opInfo.deps.getWait())
        {
            logMsg += to_string(wait) + " ";
        }
        logMsg += " update=";
        for (auto update : opInfo.deps.getUpdate())
        {
            logMsg += to_string(update) + " ";
        }
        logger.log(mv::Logger::MessageType::Debug, logMsg);
        logger.log(mv::Logger::MessageType::Debug, "Latency prediction=" + to_string(opInfo.latencyNS));

        opInfo.startNS = 0;
        for (auto depWait : opInfo.deps.getWait())
        {
            opInfo.startNS = std::max(opInfo.startNS, (*barrierInfos)[depWait].startNS);
        }
        if (opInfo.isDMA)
        {
            // Assign the DMA to a port, since our model needs to account
            // for limited port availability.
            std::size_t portIdx = 0;
            for (std::size_t portIdxT = 1; portIdxT < portLimit; ++portIdxT)
            {
                if ((*portInfos)[portIdxT].busyUntil < (*portInfos)[portIdx].busyUntil)
                {
                    portIdx = portIdxT;
                }
            }
            opInfo.portIdx = portIdx;
            opInfo.startNS = std::max(opInfo.startNS, (*portInfos)[portIdx].busyUntil);
            (*portInfos)[portIdx].busyUntil = opInfo.completeNS();
            logger.log(mv::Logger::MessageType::Debug, "Assigned port=" + to_string(portIdx));
        }
        logger.log(mv::Logger::MessageType::Debug, "Op startNS=" + to_string(opInfo.startNS));
        logger.log(mv::Logger::MessageType::Debug, "Op completeNS=" + to_string(opInfo.completeNS()));

        for (auto depUpdate : opInfo.deps.getUpdate())
        {
            (*barrierInfos)[depUpdate].startNS = std::max((*barrierInfos)[depUpdate].startNS, opInfo.completeNS());
        }
        overallLatency = std::max(opInfo.completeNS(), overallLatency);
    }

    return overallLatency;
}

// Recomputes the slack available for each port for each barrier.
// This is the amount of time between when the port becomes ready and
// the start time of the earliest op that depends on the barrier
// becoming ready.
//
// TODO: This should also take into account the next operation that
// uses the port, since moving the current tensor to CSRAM might
// make that port available earlier.
void computePortSlack(const mv::LogSender& logger,
                      std::vector<BarrierInfo>* barrierInfos,
                      std::size_t portLimit)
{
    logger.log(mv::Logger::MessageType::Debug, "Computing slack");
  
    // Compute per-barrier/per-port slack latencies.
    for (size_t barrierIdx = 0; barrierIdx < barrierInfos->size(); ++barrierIdx)
    {
        auto& barrierInfo = (*barrierInfos)[barrierIdx];
        logger.log(mv::Logger::MessageType::Debug,
                   "Considering barrier=" + to_string(barrierIdx)
                   + "; startNS=" + to_string(barrierInfo.startNS));
        if (!barrierInfo.waitOpInfos.size())
        {
            barrierInfo.earliestNS = barrierInfo.startNS;
        }
        else
        {
            auto waitIt = barrierInfo.waitOpInfos.begin();
            barrierInfo.earliestNS = (*waitIt)->startNS;
            ++waitIt;
            while (waitIt != barrierInfo.waitOpInfos.end())
            {
                barrierInfo.earliestNS = std::min(barrierInfo.earliestNS, (*waitIt)->startNS);
                ++waitIt;
            }
        }
        logger.log(mv::Logger::MessageType::Debug,
                   "Considering barrier=" + to_string(barrierIdx)
                   + "; earliestNS=" + to_string(barrierInfo.earliestNS));

        for (std::size_t portIdx = 0; portIdx < portLimit; ++portIdx)
        {
            logger.log(mv::Logger::MessageType::Debug, "Considering portIdx=" + to_string(portIdx));
            std::uint64_t portCompleteNS = 0;
            for (auto opInfo : barrierInfo.updateOpInfos)
            {
                if (opInfo->isDMA && opInfo->portIdx == portIdx)
                {
                    portCompleteNS = std::max(portCompleteNS, opInfo->completeNS());
                    logger.log(mv::Logger::MessageType::Debug, "Updated port completion to " + to_string(portCompleteNS));
                }
            }

            logger.log(mv::Logger::MessageType::Debug,
                       "Port[" + to_string(portIdx) + "].slack="
                       + to_string(barrierInfo.earliestNS - portCompleteNS));

            barrierInfo.portSlackNS[portIdx] = barrierInfo.earliestNS - portCompleteNS;
        }
    }
}

struct TensorInfo
{
    TensorInfo(mv::Tensor* tensor_, std::uint64_t size_) : tensor{tensor_}, size{size_} {}

    mv::Tensor* tensor;
    std::uint64_t size;
    std::list<OpInfo*> readers = {};
    bool updatedGraphfileIndex = false;
};

struct Benefit
{
    TensorInfo* tensorInfo;
    std::uint64_t benefitNS;
    std::uint64_t slackNS;
};

// Returns true iff rhs is a "higher benefit" than lhs.
bool operator<(const Benefit& lhs, const Benefit& rhs)
{
    return (!lhs.tensorInfo
            || (rhs.benefitNS > lhs.benefitNS)
            || (lhs.benefitNS == 0 && rhs.slackNS < lhs.slackNS));
}

// Computes the overall benefit of moving a tensor to CSRAM.
Benefit computeBenefit(const mv::LogSender& logger,
                       TensorInfo* tensorInfo,
                       std::vector<BarrierInfo>* barrierInfos,
                       std::vector<PortInfo>* portInfos,
                       std::uint64_t overallLatency)
{
    auto benefit = Benefit{tensorInfo, 0, overallLatency};

    logger.log(mv::Logger::MessageType::Debug,
               "Computing benefit; initial slack=" + to_string(benefit.slackNS));

    for (auto* opInfo : tensorInfo->readers)
    {
        for (auto& portInfo : *portInfos)
        {
            portInfo = PortInfo{0};
        }

        logger.log(mv::Logger::MessageType::Debug,
                   "Looking at reader Op=" + opInfo->op->getName()
                   + " ty=" + opInfo->op->getOpType()
                   + ": latency=" + to_string(opInfo->latencyNS)
                   + " csramNS=" + to_string(opInfo->csramNS)
                   + " size=" + to_string(opInfo->size)
                   + " port=" + to_string(opInfo->portIdx));

        for (auto depUpdate : opInfo->deps.getUpdate())
        {
            std::uint64_t newStartNS = 0;
            auto& barrierInfo = (*barrierInfos)[depUpdate];
            for (auto opInfoT : barrierInfo.updateOpInfos)
            {
                std::uint64_t completeNS = (opInfoT == opInfo
                                            ? opInfoT->completeNSUsingCSRAM()
                                            : opInfoT->completeNS());
                if (opInfoT->isDMA)
                {
                    (*portInfos)[opInfoT->portIdx].busyUntil = std::max((*portInfos)[opInfoT->portIdx].busyUntil, completeNS);
                    logger.log(mv::Logger::MessageType::Debug,
                               "Port " + to_string(opInfoT->portIdx) + " barrier slack=" + to_string(barrierInfo.portSlackNS[opInfoT->portIdx]));
                    benefit.slackNS = std::min(barrierInfo.portSlackNS[opInfoT->portIdx], benefit.slackNS);
                }
                else
                {
                    newStartNS = std::max(newStartNS, completeNS);
                }
            }

            if (barrierInfo.earliestNS != barrierInfo.startNS)
            {
                // Moving this barrier earlier isn't going to help any
                // downstream operations; we only want to consider the
                // slack for this operation.
                continue;
            }

            // TODO: This is the point where it might be useful to
            // consider additional optimization possibilities -- e.g. if
            // two DMAs are holding up the barrier, moving either to CSRAM
            // will not help, but moving both might help substantially.
            //
            // One way to do this might be to consider whether the current
            // operation's port dominates the barrier time; if it does,
            // but the benefit of moving the current DMA to come from
            // CSRAM isn't fully realized due to another port, we might
            // consider scanning the other ports for combinations of
            // tensors to move.
            for (auto& portInfo : *portInfos)
            {
                newStartNS = std::max(newStartNS, portInfo.busyUntil);
            }

            assert(newStartNS <= barrierInfo.startNS);
            newStartNS = std::min(newStartNS, barrierInfo.startNS);  // Just being careful

            benefit.benefitNS += barrierInfo.startNS - newStartNS;
        }
    }

    return benefit;
}

// Sets the tensor (and sub-tensors) to the indicated graph file
// index, and returns the next graph file index to use for
// assignments.
unsigned setTensorIndex(const mv::LogSender& logger, unsigned numClusters, TensorInfo* ti, unsigned idx)
{
    // N.B. This is essentially duplicating the logic for
    // graphFileIndex assignment used in allocate_memory_kmb.
    if (ti->tensor->hasAttr("splitStrategy")
        && !ti->tensor->hasAttr("weightTable")
        && !ti->tensor->hasAttr("sparsityMap")
        && ti->tensor->get<std::string>("splitStrategy") == "SplitOverK")
    {
        for (unsigned j = 0; j < numClusters; ++j)
        {
            logger.log(mv::Logger::MessageType::Debug,
                       "Setting graphFileIndex (dense SplitOverK " + to_string(j)
                       + " of " + ti->tensor->getLogID() + "): " + to_string(idx));
            ti->tensor->getSubTensor(j).set<unsigned>("graphFileIndex", idx++);
        }
    }
    else
    {
        logger.log(mv::Logger::MessageType::Debug,
                   "Setting graphFileIndex (fallback " + ti->tensor->getLogID() + "): " + to_string(idx));
        ti->tensor->set<unsigned>("graphFileIndex", idx++);
    }
    ti->updatedGraphfileIndex = true;
    return idx;
}

void layoutDMAFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    auto globalConfig = model.getGlobalConfigParams();

    int csramLimit = 0;
    if (passDesc.hasAttr("csramLimit"))
    {
        csramLimit = passDesc.get<int>("csramLimit");
    }
    else if (globalConfig->hasAttr("csramLimit"))
    {
        csramLimit = globalConfig->get<int>("csramLimit");
    }
    else
    {
        pass.log(mv::Logger::MessageType::Debug, "No CSRAM available (missing CSRAM attribute)");
        return;
    }

    if (csramLimit <= 0)
    {
        pass.log(mv::Logger::MessageType::Debug, "No CSRAM available: CSRAM limit=" + to_string(csramLimit));
        return;
    }

    pass.log(mv::Logger::MessageType::Debug, "CSRAM limit=" + to_string(csramLimit));

    int portLimit = 1;
    if (passDesc.hasAttr("dmaControllers"))
    {
        portLimit = passDesc.get<int>("dmaControllers");
    }
    else if (globalConfig->hasAttr("dmaControllers"))
    {
        portLimit = globalConfig->get<int>("dmaControllers");
    }

    pass.log(mv::Logger::MessageType::Debug, "DMA Port limit=" + to_string(portLimit));

    mv::DataModel dataModel(model);
    unsigned numClusters = dataModel.getGlobalConfigParams()->get<int>("Number_of_Clusters");

    mv::ControlModel controlModel(model);
    auto ops = controlModel.schedulingSort();

    // The goal of this pass is to determine a set of weight tensors
    // to be placed into CSRAM (instead of DDR), with prioritization,
    // to reduce the overall computation latency.
    //
    // So: We have a task list in scheduler order.  Conceptually, these
    // tasks are going to be evaluated by some number of logical
    // hardware components, and we can map each task to its hardware
    // component (in fact, we don't even need to know about the hardware
    // components in advance, other than the DMA controllers).  Task
    // serialization by sequential hardware components and barriers
    // creates dependencies between tasks.
    //
    // To reduce the overall latency, we repeatedly replace weights in
    // DDR with weights in CSRAM, adjusting the corresonding weight
    // tensor placement and DMA tasks appropriately.
    //
    // To select a weight tensor to place into CSRAM: we consider each
    // weight tensor DMA operation in turn.  For each such operation, we
    // determine a Cost for moving the weight tensor to CSRAM (the size
    // of the data to DMA), and a Benefit (the reduction in overall
    // latency).  We then select the weight tensor DDR DMA task with the
    // highest benefit and a cost that fits within our remaining CSRAM
    // budget, and reassign the corresponding weight tensor from DDR to
    // CSRAM, keeping track of the order in which we move the tensors
    // (for runtime prioritization in low-resource scenarios).
    //
    // When computing the Benefit, there may be cases where there's no
    // latency benefit to be gained by moving a weight tensor to CSRAM
    // (e.g. there's overlapping compute or other DMAs on the critical
    // path), but there's still available CSRAM space for a weight
    // tensor.  In this case, we want to move additional weight
    // tensors to CSRAM, selecting tensors whose DMAs are close to
    // being on the critical path -- i.e. cases where we're low on
    // slack latency, running close to the limit.  So separately from
    // the latency benefit, we track the "slack" for each potential
    // update, and if there's no plan that offers a concrete benefit,
    // we select the plan with the least slack, resolving ties in
    // favor of the smaller weight tensor (the better to spread slack
    // latency around the overall computation).
    //
    // Note that we do not rebuild our per-barrier start times after
    // each DMA source reassignment.  The reason is that while moving
    // a weight tensor to CSRAM can affect the benefit of moving other
    // tensors, it typically won't, and incorporating that benefit
    // requires O(n^2) time, which turns out to be problematic for the
    // networks we care about, even with some work done to reduce the
    // relevant constant factors.
    //
    // There're two minor limitations to this algorithm:
    //
    //   1) Since there may be multiple DMA controllers operating in
    //      parallel, there will be cases where moving one of two
    //      weight tensors to CSRAM provides almost no benefit, but
    //      where moving both tensors provides a substantial benefit
    //      (although note that this benefit will always be at double
    //      the CSRAM cost compared to cases where moving a single
    //      tensor provides a speedup).
    //
    //      To solve this, a future version of this code might model
    //      weight tensor groups, trying to identify sets of tensors
    //      that, when moved to CSRAM together, provide a performance
    //      benefit commensurate with their cost.
    //
    //   2) There's another minor limitation: because we're
    //      considering operations in turn, and because we're
    //      unconcerned with loops, we will underestimate the impact
    //      of a weight tensor that's used multiple times.  In
    //      practice, that's not a concern for current networks we're
    //      optimizing for, but we may want to handle that case
    //      correctly in the future.

    unsigned opLimit = 0;
    unsigned barrierMax = 0;
    for (auto op : ops)
    {
        ++opLimit;
        auto deps = op->get<mv::BarrierDependencies>("BarrierDeps");
        for (auto depWait : deps.getWait())
        {
            barrierMax = std::max(barrierMax, depWait);
        }
        for (auto depUpdate : deps.getUpdate())
        {
            barrierMax = std::max(barrierMax, depUpdate);
        }
    }

    if (!barrierMax)
    {
        // Nothing to do.
        return;
    }

    std::vector<OpInfo> opInfos;
    opInfos.reserve(opLimit);

    // Gather operation infos, including latencies.
    for (auto op : ops)
    {
        auto opInfo = opInfos.emplace(opInfos.end(), analyzeOp(pass, *op));
        opInfo->deps = op->get<mv::BarrierDependencies>("BarrierDeps");
        std::string logMsg = "Found Op=" + op->getName()
          + " ty=" + op->getOpType()
          + ": wait=";
        for (auto wait : opInfo->deps.getWait())
        {
            logMsg += to_string(wait) + " ";
        }
        logMsg += " update=";
        for (auto update : opInfo->deps.getUpdate())
        {
            logMsg += to_string(update) + " ";
        }
        logMsg += " size=" + to_string(opInfo->size) + op->toString();
        pass.log(mv::Logger::MessageType::Debug, logMsg);
    }

    std::vector<BarrierInfo> barrierInfos(barrierMax+1, BarrierInfo{0, 0, {}, {}, std::vector<std::uint64_t>(portLimit, 0)});
    for (auto& opInfo : opInfos)
    {
        for (auto depWait : opInfo.deps.getWait())
        {
            barrierInfos[depWait].waitOpInfos.emplace_back(&opInfo);
        }
        for (auto depUpdate : opInfo.deps.getUpdate())
        {
            barrierInfos[depUpdate].updateOpInfos.emplace_back(&opInfo);
        }
    }

    std::vector<PortInfo> portInfos(portLimit);

    std::list<TensorInfo> tensorInfos;
    std::unordered_map<mv::Tensor*, TensorInfo*> tensorInfoMap;

    for (auto t = model.tensorBegin(); t != model.tensorEnd(); ++t)
    {
        if (!t->hasAttr("allocators") || !t->get<std::set<std::string>>("allocators").count("GraphFile"))
        {
            continue;
        }
        auto it = tensorInfos.emplace(tensorInfos.end(), TensorInfo{&*t, t->computeTotalSize()});
        tensorInfoMap[&*t] = &*it;
    }

    for (auto& opInfo : opInfos)
    {
        for (auto& tensor : opInfo.op->getInputTensor())
        {
            auto it = tensorInfoMap.find(&*tensor);
            if (it != tensorInfoMap.end())
            {
                it->second->readers.emplace_back(&opInfo);
            }
        }
    }

    std::uint64_t overallLatency = simRunModel(pass, &opInfos, &barrierInfos, &portInfos);
    computePortSlack(pass, &barrierInfos, portLimit);

    pass.log(mv::Logger::MessageType::Debug, "OverallLatency=" + to_string(overallLatency));

    // Compute the benefit for each candidate tensor movement.
    std::list<Benefit> benefits;
    for (auto& tensorInfo : tensorInfos)
    {
        auto benefit = computeBenefit(pass, &tensorInfo, &barrierInfos, &portInfos, overallLatency);

        pass.log(mv::Logger::MessageType::Debug,
                 "Benefit=" + to_string(benefit.benefitNS) + " slack=" + to_string(benefit.slackNS));
        benefits.emplace_back(std::move(benefit));
    }

    benefits.sort([](const Benefit& lhs, const Benefit& rhs)
                  {
                      // Sort s.t. the highest benefit is "smaller" ->
                      // at the front of the benefits list.
                      return !(lhs < rhs);
                  });

    std::uint64_t csramAvailable = static_cast<std::uint64_t>(csramLimit);
    int currentGraphfileIndex = 0;

    // Process benefits in two passes: one pass to pull out as many
    // tensors as will fit into the amount of CSRAM memory we think we
    // have available, and a second pass to handle the remainder.
    for (auto bit = benefits.begin(); bit != benefits.end();)
    {
        auto& benefit = *bit;
        if (benefit.tensorInfo->size <= csramAvailable)
        {
            csramAvailable -= benefit.tensorInfo->size;
            currentGraphfileIndex = setTensorIndex(pass, numClusters, benefit.tensorInfo, currentGraphfileIndex);
            for (auto updateOpInfo : benefit.tensorInfo->readers)
            {
                updateOpInfo->latencyNS = updateOpInfo->csramNS;
            }
            bit = benefits.erase(bit);
        }
        else
        {
            ++bit;
        }
    }

    // Second pass, applying the remaining benefits (=> whatever
    // didn't fit into CSRAM) in order, in case there actually is
    // CSRAM available at runtime.
    for (auto& benefit : benefits)
    {
        currentGraphfileIndex = setTensorIndex(pass, numClusters, benefit.tensorInfo, currentGraphfileIndex);
    }

    // Recompute port assignments based on updated tensor placements.
    simRunModel(pass, &opInfos, &barrierInfos, &portInfos);

    // At this point, we have pretty good port assignments for all
    // operations in the opInfos vector, so update the actual operations.
    for (auto& opInfo : opInfos)
    {
        if (opInfo.isDMA)
        {
            opInfo.op->set<std::uint8_t>("port", opInfo.portIdx);
        }
    }

    // Rewrite remaining graphFileIndex values.
    for (auto& ti : tensorInfos)
    {
        if (!ti.updatedGraphfileIndex)
        {
            currentGraphfileIndex = setTensorIndex(pass, numClusters, &ti, currentGraphfileIndex);
        }
    }
}
