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

    auto targetTotalCmx = target.memoryDefs().at("VPU_CMX_NN").size;
    auto cmxUser = globalConfig->hasAttr("NNCMXPerSlice") ? globalConfig->get<int>("NNCMXPerSlice") : targetTotalCmx;
    auto cmx = std::min(targetTotalCmx, cmxUser);

    auto targetTotalClusters = target.nceDefs().at("Clusters").totalNumber;
    auto clustersUser = globalConfig->hasAttr("Number_of_Clusters") ? globalConfig->get<int>("Number_of_Clusters") : targetTotalClusters;
    auto clusters = std::min(targetTotalClusters, clustersUser);

    auto memoryHack = globalConfig->hasAttr("MemoryHack") && globalConfig->get<bool>("MemoryHack");
    auto safetyFactor = globalConfig->hasAttr("CMX_memory_overflow_safety_factor") ? globalConfig->get<double>("CMX_memory_overflow_safety_factor") : 0.9;

    if (memoryHack)
        cmx *= safetyFactor;
    else
    {
        auto cmxPerCluster = cmx / targetTotalClusters;
        cmxPerCluster *= safetyFactor;
        cmx = cmxPerCluster * clusters;
    }

    globalConfig->set<unsigned>("cmx", cmx);
    globalConfig->set<unsigned>("clusters", clusters);
}
