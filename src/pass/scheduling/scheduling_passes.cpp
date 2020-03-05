#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/target/kmb/workloads.hpp"
#include <cmath>
#include <numeric>
#include <cmath>
#include <unordered_set>

static void generateSchedulingFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void barrierIndexAssignmentFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateBarrierRefsFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void storeBarriersNamesInTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateCountsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void hackExecutionScheduleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&);
static void correctExecutionScheduleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&);
static void reorderDmasInScheduleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&);

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
            barrierRef.setWaitBarrier(barrier.getIndex());
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
