#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <limits>

static void buildGraphFileKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);
static void updatePhysicalIDinGraphFileFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&);
static void generateBlobKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);
static void DMAOrderingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);
namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(BuildGraphFileKmb)
        .setFunc(buildGraphFileKmbFcn)
        .setDescription("Builds the graphfile according to the schema");
    }

    namespace pass
    {
        MV_REGISTER_PASS(UpdatePhysicalIDinGraphFile)
        .setFunc(updatePhysicalIDinGraphFileFcn)
        .setDescription("Updates physical barrier id to task dependency in graphfile");
    }
    
    namespace pass
    {
        MV_REGISTER_PASS(GenerateBlobKmb)
        .setFunc(generateBlobKmbFcn)
        .setDescription("Dumps the graphfile to disk as an executable blob file for KMB");
    }

    namespace pass
    {
        MV_REGISTER_PASS(DMAOrdering)
        .setFunc(DMAOrderingFcn)
        .setDescription("This pass stores an attribute (DMA-level, DPU-schedule-number) on all DMAs so that they can be serialized in the correct order");
    }
}

void buildGraphFileKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&)
{   
    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE)
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(td);
    rm.buildGraphFile(model, td, passDesc);
}

void updatePhysicalIDinGraphFileFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&){
    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE)
    bool isStatic= false;
    auto globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("enableStaticBarriers"))
    {
        isStatic= globalParams->get<bool>("enableStaticBarriers");
    }
    if (!isStatic){
        return;
    }
    
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(td);
    const MVCNN::GraphFileT& gf= rm.getGraphFile();
    const auto& barrierTasks= gf.task_lists[2]->content;
    std::unordered_map<unsigned , unsigned> virtual2real;
    // construct mapping: {vid->pid}
    for (unsigned vid=0; vid<barrierTasks.size(); vid++){
        const MVCNN::ControllerTaskT* bt= static_cast<const MVCNN::ControllerTaskT*>(barrierTasks[vid]->task.value);
        const MVCNN::BarrierConfigurationTaskT* bt2=
                static_cast<const MVCNN::BarrierConfigurationTaskT*>(bt->task.value);
        virtual2real.insert(std::make_pair(vid, bt2->target->barrier_id));
    }
    
    auto& computeTasks= gf.task_lists[0]->content;
    auto& dmaTasks= gf.task_lists[1]->content;
    auto update_physical_id = [&virtual2real](vector<unique_ptr<MVCNN::TaskT>>& tasks){
        for (auto& task : tasks){
            auto& virtual_wait= task->associated_barriers->virtual_wait_barriers;
            auto& virtual_update= task->associated_barriers->virtual_update_barriers;
            vector<uint32_t> physical_wait, physical_update;
            for (auto vid : virtual_wait){
                auto itr= virtual2real.find(vid);
                if (itr == virtual2real.end()){
                   throw std::runtime_error("updatePhysicalIDinGraphFileFcn: Invalid wait virtual ID in task " + task->name); 
                }
                physical_wait.push_back(itr->second);
            }
            for (auto vid : virtual_update){
                auto itr= virtual2real.find(vid);
                if (itr == virtual2real.end()){
                   throw std::runtime_error("updatePhysicalIDinGraphFileFcn: Invalid update virtual ID in task " + task->name); 
                }
                physical_update.push_back(itr->second);
            }
            task->associated_barriers->wait_barriers= physical_wait;
            task->associated_barriers->update_barriers= physical_update;
        }
    };
    update_physical_id(computeTasks);
    update_physical_id(dmaTasks);
}

void generateBlobKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&)
{   
    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE)
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(td);

    if (passDesc.hasAttr("metaInfoSerializer"))
    {
        std::function<void(MVCNN::GraphFileT&)> metaInfoFcn =
            passDesc.get<std::function<void(MVCNN::GraphFileT&)>>("metaInfoSerializer");
        rm.serializeHelper(metaInfoFcn);
    }
    if (passDesc.hasAttr("output")) // if attribute missing, blob file not written
    {
        auto output = passDesc.get<std::string>("output");
        mv::utils::validatePath(output);

        rm.serialize(output);
    }
}

mv::Data::OpListIterator findChildDPUorUPATaskOp(mv::ComputationModel& model, mv::Data::OpChildIterator& op)
{
    mv::OpModel om(model); 
    mv::Data::OpListIterator childOp = om.getOp(op.leftmostChild()->getName());
   
    if(childOp->getOpType() == "Output")
            return om.getOutput();

    if(childOp->getOpType() == "DPUTask" || childOp->getOpType() == "UPATask")
            return om.getOp(childOp->getName());

    while(!(childOp->getOpType() == "DPUTask") && !(childOp->getOpType() == "UPATask"))
    { 
        

        childOp = om.getOp(childOp.leftmostChild()->getName());
        if(childOp->getOpType() == "DPUTask" || childOp->getOpType() == "UPATask")
            return om.getOp(childOp->getName());
        if(childOp->getOpType() == "Output")
            return om.getOutput();
        else 
            childOp = om.getOp(childOp.leftmostChild()->getName());

        if(childOp->getOpType() == "Output")
            return om.getOutput();
    } 
    return om.getOp(childOp->getName());
}

static void DMAOrderingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& /*td*/, mv::Element& /*passDesc*/, mv::Element&)
{ 
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS) 
    mv::OpModel om(model);
    mv::ControlModel cm(model); 
    mv::Data::OpListIterator dputask; 
    unsigned dpuTaskschedulingNumber = 0;
    unsigned dpulevel = 0;
    auto sortedOps = cm.topologicalSort();
    std::vector<unsigned> dpuTaskschedulingNumbers;
    unsigned maxdpuTaskschedulingNumber = 0;

    auto dmas = om.getOps("DMATask");

    for(auto& dmaOp: dmas) {

            dpuTaskschedulingNumbers.clear();
            for(auto son = dmaOp.leftmostChild(); son != om.opEnd(); ++son)
            {
                auto task = om.getOp(son->getName());
                if(task->getOpType() != "DPUTask" && (task->getOpType() != "Output" && task->getOpType() != "UPATask"))
                    task = findChildDPUorUPATaskOp(model, son);
       
                if(task->hasAttr("schedulingNumber"))
                {
                    dpuTaskschedulingNumber = task->get<unsigned>("schedulingNumber");
                    dpuTaskschedulingNumbers.push_back(dpuTaskschedulingNumber);
                    dpulevel = task->get<unsigned>("layerNumber");
                }
               
                auto dmaTasklayernumber = dmaOp->get<unsigned>("layerNumber");

                if(dpuTaskschedulingNumber > maxdpuTaskschedulingNumber)
                    maxdpuTaskschedulingNumber = dpuTaskschedulingNumber;


                if(task->getOpType() == "Output")
                {
                    maxdpuTaskschedulingNumber = std::numeric_limits<unsigned>::max();
                    dpuTaskschedulingNumber= maxdpuTaskschedulingNumber;
                    dpuTaskschedulingNumbers.push_back(dpuTaskschedulingNumber);
                }
               
                dmaOp->set<unsigned>("DPULevel",dpulevel);
                dmaOp->set<unsigned>("DMALevel",dmaTasklayernumber);
                dmaOp->set<unsigned>("DPU-schedule-number",*std::min_element(std::begin(dpuTaskschedulingNumbers), std::end(dpuTaskschedulingNumbers)));
                dmaOp->set<std::array<unsigned short, 2>>("DMALevel-DPU-schedule-number", {
                    static_cast<unsigned short>(dmaTasklayernumber),
                    static_cast<unsigned short>(dpuTaskschedulingNumber)
                  });
            }
    }
}
