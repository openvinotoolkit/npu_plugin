#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include <math.h>

static void computeMemoryFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);

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

static void computeMemoryFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& compilationDescriptor, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    auto globalConfig = model.getGlobalConfigParams();


    // ASSUMPTION: One cluster if not specified
    auto targetTotalClusters = static_cast<int>(target.nceDefs().at("Clusters").totalNumber);
    auto clustersUser = globalConfig->hasAttr("Number_of_Clusters") ? globalConfig->get<int>("Number_of_Clusters") : 1;
    auto clusters = std::min(targetTotalClusters, clustersUser);

    // ASSUMPTION: User always uses full memory
    auto cmx = target.memoryDefs().at("VPU_CMX_NN").size;
    // ASSUMPTION: Except for when we use the memory hack, the amount of cmx available refers to the amount available to a single cluster
    // This means that subtensor have to be used for the computation of memory requirements during maxcut.
    // Also, for now memory hack makes sense just for 1 cluster.
    auto cmxPerCluster = cmx / targetTotalClusters;

    // ASSUMPTION: Default memory safety factor is 0.925
    auto safetyFactor = globalConfig->hasAttr("CMXOverflowSafetyFactor") ? globalConfig->get<double>("CMXOverflowSafetyFactor") : 0.925;

    if(globalConfig->hasAttr("cmx"))
    {
        //Bypass safety factor
        unsigned userMemory = globalConfig->get<int>("cmx");
        if(userMemory <= cmxPerCluster)
            cmx = userMemory;
    }

    auto updatedCmx = cmx * safetyFactor;

    globalConfig->set<unsigned>("cmx", updatedCmx); // this has cmx including safety factor
    globalConfig->set<unsigned>("totalCmx", cmx); // use the lower value of the two total CMX - target vs user requested
    globalConfig->set<unsigned>("clusters", clusters);
}
