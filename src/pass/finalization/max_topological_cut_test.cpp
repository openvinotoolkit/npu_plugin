#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/koala.hpp"
#include <iostream>

static void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MaxTopologicalCutAndPartialSerialisation)
        .setFunc(maxTopologicalCutAndPartialSerialisationPass)
        .setDescription(
            "Perform the max topological cut algorithm and partial serialisation (if required) to schedule the DAG."
        );
    }

}


void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::json::Object&)
{
    mv::KoalaClass flowGraph;
    
    /*Convert to MCM graph to KOALA graph*/
    flowGraph.convertMcMGraphToKoalaGraph(pass, model);

    // /*Calculate max topological cut and get the cut edges*/
    auto maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
   
    // /*New edges added to the graph from partial serialisation, these will be added to the McM graph*/
    // std::vector<koalaGraph::PEdge> partialSerialisationEdgesAdded;

    /*Get CMX memory*/
    auto memDefs = target.memoryDefs();
    auto availableNNCMX = memDefs.find("VPU_CMX_NN")->second.size;

    // /*Get the number of clusters that the VPU supports*/
    auto nceDefs = target.nceDefs();
    auto numberOfVPUClusters = nceDefs.find("Clusters")->second.totalNumber;

    /*Get the CMX safety factor*/
    std::shared_ptr<mv::Element> returnedParams = model.getGlobalConfigParams();
    double cmxSafetyFactor = returnedParams->get<double>("CMX_memory_overflow_safety_factor");

    /*Note available CMX memory is 3760128 /number of supported VPU clusters (always 4)*/
    auto cmxMemory = (availableNNCMX / numberOfVPUClusters) * cmxSafetyFactor;

    /*Repeat partial serialisation until max topological cut is less than CMX memory*/
    while (maxTopologicalCut.first > cmxMemory) {
        int res = flowGraph.performPartialSerialisation(pass, maxTopologicalCut.first, maxTopologicalCut.second);
        if(!res)
            maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
        else
            throw std::runtime_error("Unable to find partial serialisation edge, exit");
       
    }

    // /*Add the partial serialisaion edges to the mcmGraph*/
    // insertpartialSerialisationEdgesInMcmGraph(model, partialSerialisationEdgesAdded);
}
