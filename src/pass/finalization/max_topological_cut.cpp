#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/koala_graph_scheduler.hpp"
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
    // mv::ControlModel controlModel(model);
    // auto topologicallySortedOps = controlModel.topologicalSort();

    // auto removeOps = [] (std::vector<mv::Control::OpListIterator>& list, const std::string& opType)
    // {
    //     list.erase(std::remove_if(list.begin(), list.end(), [opType](mv::Control::OpListIterator it) { return it->getOpType() == opType;}), list.end());
    // };

    // removeOps(topologicallySortedOps, "Constant");
    // removeOps(topologicallySortedOps, "ConstantInt");
    // removeOps(topologicallySortedOps, "ConstantDataElement");

    // for(auto vecIt = topologicallySortedOps.begin(); vecIt != topologicallySortedOps.end(); ++vecIt)
    // {
    //     //if ((*vecIt)->getOpType() == "DMATask" || (*vecIt)->getOpType() == "Deallocate" || (*vecIt)->getOpType() == "Input" || (*vecIt)->getOpType() == "DPUTask")
    //     std::cout << (*vecIt)->getName() << " : " << (*vecIt).inputsSize() << " : " << (*vecIt).outputsSize()  << std::endl;
    // }

    int networkMemoryRequirement;
    double percentageMemory; 
    mv::KoalaGraphScheduler flowGraph;
    
    /*Convert to MCM graph to KOALA graph*/
    flowGraph.convertMcMGraphToKoalaGraph(pass, model);

    /*Calculate max topological cut and get the cut edges*/
    auto maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
   
    auto returnedParams = model.getGlobalConfigParams();
    double cmxMemory = returnedParams->get<unsigned>("cmx");

    networkMemoryRequirement = maxTopologicalCut.first / 1024;
    percentageMemory = (maxTopologicalCut.first / cmxMemory) * 100.00;

    pass.log(mv::Logger::MessageType::Info, "The network requires " + std::to_string(networkMemoryRequirement) + " kB of available CMX memory " + std::to_string(percentageMemory) + "%");
    
    /*Repeat partial serialisation until max topological cut is less than CMX memory*/
    while (maxTopologicalCut.first > cmxMemory)
    {
        flowGraph.performPartialSerialisation(pass, maxTopologicalCut.second);
        maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
        networkMemoryRequirement = maxTopologicalCut.first / 1024;
        percentageMemory = (maxTopologicalCut.first / cmxMemory) * 100;
        pass.log(mv::Logger::MessageType::Info, "The network requires " + std::to_string(networkMemoryRequirement) + " kB of available CMX memory " + std::to_string(percentageMemory) + "%");
    }
    /*Add the partial serialisaion edges to the mcmGraph*/
    flowGraph.insertpartialSerialisationEdgesInMcmGraph(model);
}
