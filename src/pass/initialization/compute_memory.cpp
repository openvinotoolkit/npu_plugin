#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include <math.h>

static void computeMemoryFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(ComputeMemory)
        .setFunc(computeMemoryFcn)
        .setDescription(
            "Computes the correct amounts of memory and sets the into the global configuration parameters"
        );
    }
}

static void computeMemoryFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& compilationDescriptor, mv::json::Object&)
{
    auto globalConfig = model.getGlobalConfigParams();

    // ASSUMPTION: User always uses full memory
    auto cmx = target.memoryDefs().at("VPU_CMX_NN").size;

    // ASSUMPTION: One cluster if not specified
    auto targetTotalClusters = static_cast<int>(target.nceDefs().at("Clusters").totalNumber);
    auto clustersUser = globalConfig->hasAttr("Number_of_Clusters") ? globalConfig->get<int>("Number_of_Clusters") : 1;
    auto clusters = std::min(targetTotalClusters, clustersUser);

    // ASSUMPTION: By default there is no memory hack
    auto memoryHack = globalConfig->hasAttr("MemoryHack") && globalConfig->get<bool>("MemoryHack");

    // ASSUMPTION: Default memory safety factor is 0.925
    auto safetyFactor = globalConfig->hasAttr("CMX_memory_overflow_safety_factor") ? globalConfig->get<double>("CMX_memory_overflow_safety_factor") : 0.925;

    // ASSUMPTION: Except for when we use the memory hack, the amount of cmx available refers to the amount available to a single cluster
    // This means that subtensor have to be used for the computation of memory requirements during maxcut.
    // Also, for now memory hack makes sense just for 1 cluster.
    if (memoryHack)
        cmx *= safetyFactor;
    else
    {
        auto cmxPerCluster = cmx / targetTotalClusters;
        cmx = cmxPerCluster * safetyFactor;

        if(globalConfig->hasAttr("cmx"))
        {
            unsigned userMemory = globalConfig->get<int>("cmx");
            if(userMemory <= cmxPerCluster)
                cmx = userMemory;
        }
    }

    globalConfig->set<unsigned>("cmx", cmx);
    globalConfig->set<unsigned>("clusters", clusters);
}
