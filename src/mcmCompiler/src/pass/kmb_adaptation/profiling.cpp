#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/computation/flow/implicit_flow.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"

#include "pass/lp_scheduler/barrier_scheduler_pass.hpp"

#define HW_TIMER_ABSOLUTE_ADDR  0x203300BC  

static void AddTaskProfilingDMAFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AddDPUTasksProfilingDMA)
            .setFunc(AddTaskProfilingDMAFcn)
            .setDescription(
               "Add DMA Tasks for DPU Tasks profiling");
    }
}

void allocateImplicitOperationsOp(mv::Data::OpListIterator opIterator, mv::Control::StageIterator stageIt, const mv::pass::PassEntry& pass,
                                  mv::ComputationModel& model);
void insertBarriersIntoControlFlowGraph(mv::ComputationModel& model, const mv::Element& passDesc, const std::vector<mv::Barrier>& barriers);


/* In order to correctly calculate the layer timing in case of parallel execution we need to find the parent task 
   and calculate timings between current task value and parent one
   here we recursively going thought all parent branches and looks for the task with the closest layerNumber to the current one */ 
static mv::Control::OpListIterator findParentDPUorUPATask(mv::Control::OpListIterator opIterator, mv::ControlModel& cm) {
    auto ret = cm.opEnd();
    unsigned retLayerNumber = 0;
    for (auto parentOp = opIterator.leftmostParent(); parentOp != cm.opEnd(); ++parentOp) {
        if ((parentOp->getOpType() == "DPUTask") || (parentOp->getOpType() == "UPATask")) {
            return parentOp;
        }
        auto op = findParentDPUorUPATask(parentOp, cm);
        if (op == cm.opEnd()) continue; 
        auto opParentLayerNumber = op->get<unsigned>("layerNumber");

        /* The closest parent is found -> return it */
        if (opIterator->hasAttr("layerNumber") && (opParentLayerNumber+1 == opIterator->get<unsigned>("layerNumber")))
            return op;

        /* Compare all recursion result and return the closest node */
        if (opParentLayerNumber > retLayerNumber) {
            ret = op;
            retLayerNumber = opParentLayerNumber;
        }
    }

    return ret;
}

void AddDMAtoBarrier(mv::Control::OpListIterator barrierOp, std::string dmaName, unsigned layerNumber, unsigned parentDMA, unsigned opId, mv::ComputationModel& model, mv::Data::TensorIterator &profilingTensor, std::vector<mv::Data::TensorIterator> &profilingDmas)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto& barrier = barrierOp->get<mv::Barrier>("Barrier");
    auto barrierName = barrierOp->getName();

    auto profilingDma = om.dMATask(dmaName+"_"+to_string(parentDMA)+"_"+to_string(layerNumber), profilingTensor, mv::DmaDirectionEnum::HW2DDR, 0);
    profilingDmas.push_back(profilingDma);

    auto profilingDmaOp = om.getSourceOp(profilingDma);
    profilingDmaOp->set<unsigned>("opId", opId);
    profilingDmaOp->set<unsigned>("layerNumber", layerNumber);
    //schedulingNumber is always 0 as this DMA has the higist prio under the same barrier
    profilingDmaOp->set<unsigned>("DPU-schedule-number", 0);
    profilingDmaOp->set<unsigned>("schedulingNumber", 0);
    profilingDmaOp->set("BarrierDeps", mv::BarrierDependencies());
    auto& barrierRef = profilingDmaOp->get<mv::BarrierDependencies>("BarrierDeps");
    barrierRef.addWaitBarrier(barrier.getIndex());

    barrier.addConsumer(dmaName);

    auto barrierDataOp = om.getOp(barrierName);
    cm.defineFlow(barrierDataOp, profilingDmaOp);
}

/* The expectation is to have after each DPU/UPA task barrier with oply one producer(corresponding task) 
 * this function find this barrier */
static mv::Control::OpListIterator FindSuitableBarrier(mv::Control::OpListIterator opIt, mv::ControlModel &cm, const mv::pass::PassEntry& pass)
{
    int minProducersCount = std::numeric_limits<int>::max();
    mv::Control::OpListIterator retOp = cm.opEnd();
    for(auto childOp = opIt.leftmostChild(); childOp != cm.opEnd(); ++childOp) {
        if (childOp->getOpType() == "BarrierTask") {
            auto& barrier = childOp->get<mv::Barrier>("Barrier");
            if (barrier.getNumProducers() < minProducersCount) {
                minProducersCount = barrier.getNumProducers();
                retOp = childOp;   
            }
        }
    }

    if ((retOp != cm.opEnd()) && (minProducersCount != 1)) {
        pass.log(mv::Logger::MessageType::Warning, "There is no barrier with 1 producer for " + opIt->getName());
    }

    return retOp;
}

// Pass role: Add DMA Tasks for DPU task profiling (if needed).
void AddTaskProfilingDMAFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element &)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    /* Check if profiling DMAs should be added */ 
    bool enable_profiling = false;
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("PerformanceCounting") && globalParams->get<bool>("PerformanceCounting"))
        enable_profiling = true;
    if (passDesc.hasAttr("force_profiling") && passDesc.get<bool>("force_profiling"))
        enable_profiling = true;

    if (!enable_profiling) return;

    /* Specify here the Free Running counter address */
    auto profilingTensor = om.constantInt("profilingInput", {HW_TIMER_ABSOLUTE_ADDR}, {1,1,1,1}, mv::DType("Int32"), mv::Order("NCHW"));
    /* Mandatory fields */
    std::set<std::string> toSet;
    profilingTensor->set<std::set<std::string>>("flows", toSet);
    profilingTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
    /* Get last_graphFileIndex to be able properly allocate address constant */
    auto graphFileIndex = dm.getGlobalConfigParams()->get<int>("last_graphFileIndex");
    profilingTensor->set<unsigned>("graphFileIndex", graphFileIndex);
    graphFileIndex++;
    dm.getGlobalConfigParams()->set<int>("last_graphFileIndex", graphFileIndex);

    std::vector<mv::Data::TensorIterator> profilingDmas;
    std::map<std::string, int> TaskDMAMap;

    for(auto opIt:cm.schedulingSortDPUorUPA()) {
        auto barrierOp = FindSuitableBarrier(opIt, cm, pass);
        bool lastTask = false;

        if (barrierOp == cm.opEnd()) {
            std::vector<mv::Barrier> barriers;
            std::set<std::string> producers, consumers;
            producers.insert(opIt->getName());
            mv::Barrier new_barrier(producers, consumers);
            barriers.push_back(new_barrier);
            insertBarriersIntoControlFlowGraph(model, passDesc, barriers);
            mv::lp_scheduler::Control_Model_Barrier_Scheduler::renumberBarrierTasks(om);
            barrierOp = opIt.leftmostChild();
            
            /* Add new barrier dependency */
            auto& barrierRef = opIt->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.addUpdateBarrier(barrierOp->get<mv::Barrier>("Barrier").getIndex());

            /* Remove trailing tag as we will add barrier dependency */
            if (opIt->hasAttr("trailing")) 
                opIt->set<bool>("trailing", false);
            lastTask = true;
        }

        if (barrierOp != cm.opEnd()) {
            auto barrierName = barrierOp->getName();
            auto dmaName = opIt->getName();
            auto opId = opIt->get<unsigned>("opId");
            auto layerNumber = opIt->get<unsigned>("layerNumber");
            
            auto parentDPU = findParentDPUorUPATask(opIt, cm);
            if (parentDPU == cm.opEnd()) {
                /* Add DMA to the first parent barrier */
                for(auto parentOp = opIt.leftmostParent(); parentOp != cm.opEnd(); ++parentOp) {
                    if (parentOp->getOpType() == "BarrierTask") {
                        AddDMAtoBarrier(parentOp, dmaName+"_PROFBEGIN", 1, 0, opId, model, profilingTensor, profilingDmas);
                        TaskDMAMap[parentOp->getName()] = profilingDmas.size()-1;
                        break;
                    }
                }
            }

            unsigned parentDMA = 0;
            if (parentDPU != cm.opEnd()) {
                auto parentDMAit = TaskDMAMap.find(parentDPU->getName());
                if (parentDMAit == TaskDMAMap.end() ) {
                throw std::logic_error("Cannot find parent DMA for the DPU");
                } else parentDMA = parentDMAit->second;
            }

            layerNumber++;
            dmaName += (!lastTask) ? "_PROFMIDDLE" : "_PROFEND";
            AddDMAtoBarrier(barrierOp, dmaName, layerNumber, parentDMA, opId, model, profilingTensor, profilingDmas);
            TaskDMAMap[opIt->getName()] = profilingDmas.size()-1;
        }
    }

    if (!profilingDmas.size()) {
        throw std::logic_error("No profiling DMAs added during Task loop");
        return;
    }
    /*
     * Combine all DMAs by creating implicitConcat
     */
    auto ProfilingConcat = om.implicitConcat("ProfilingConcat", profilingDmas);
    //Perform ResolveImplicitOperations pass manually
    auto ProfilingConcatOp = om.getSourceOp(ProfilingConcat);
    ProfilingConcatOp->set<mv::ImplicitFlow>("ImplicitFlow", mv::ImplicitFlow(mv::ImplicitFlow::INPUT_IN_OUTPUT));
    ProfilingConcatOp->get<mv::ImplicitFlow>("ImplicitFlow").resolve();

    /* 
     * Memory allocation 
     */
    //Set locations of Tensors (UpdateImplicitLayersLocation pass)
    //OUTPUT memory has to be allocated manually
    auto stageIt = cm.getStage(0);
    profilingTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
    dm.allocateTensor("GraphFile", stageIt, profilingTensor);
    auto OutputLocation = mv::Tensor::MemoryLocation::OUTPUT;
    for (auto profilingDma: profilingDmas) {
        /* This set the "locale": "ProgrammableOutput" in blob */
        profilingDma->set<mv::Tensor::MemoryLocation>("Location", OutputLocation);
        dm.allocateTensor("ProgrammableOutput", stageIt, profilingDma);
    }
    ProfilingConcat->set<mv::Tensor::MemoryLocation>("Location", OutputLocation);
    dm.allocateTensor("ProgrammableOutput", stageIt, ProfilingConcat);

    /* 
     * Create additional output from network 
     */
    auto profilingOutput = om.output("profilingOutput", ProfilingConcat, {"Default"}, true);
    auto profilingOutputTensor = dm.defineTensor("profilingOutput", ProfilingConcat->getShape(), ProfilingConcat->getDType(), ProfilingConcat->getOrder());
    ProfilingConcat->set<uint8_t>("outputIndex", om.getNetworkOutputs().size()-1);
    /* 
     * Execute postprocessing passes 
     */
    allocateImplicitOperationsOp(ProfilingConcatOp, stageIt, pass, model);

#if 1
    //Check if network still pass the schedule check, TODO: dynamicaly get real barriers count
    auto success = mv::lp_scheduler::Control_Model_Barrier_Checker::check_schedule(cm, 32);
    printf("[BarrierSimulatorCheckPass]: %s\n", success ? "PASSED" : "FAILED"); 
#endif

    /* Debug info */
    pass.log(mv::Logger::MessageType::Info,  "Total Profiling output size: " + std::to_string(ProfilingConcat->size()));
}