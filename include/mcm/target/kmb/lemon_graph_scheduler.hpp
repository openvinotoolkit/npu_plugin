#ifndef MV_LEMONGRAPHSCHEDULER
#define MV_LEMONGRAPHSCHEDULER

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include <lemon/list_graph.h>

namespace mv
{
    /*Lemon Node info*/
    struct nodeDescription 
    {
        int id; 
        std::string name;
        std::string opType;
        int cost; //Required for optimal partial serialisation (minimises increase in the critical path)
        bool sourceNode;
        bool sinkNode;

        nodeDescription(int id=0, std::string aname = "", std::string inOpType = "", int cost = 0, bool sourcenode = false, bool sinknode = false) :
            id(id), name(aname), opType(inOpType), cost(cost), sourceNode(sourcenode), sinkNode(sinknode) { }
    };

    /*Lemon Edge info*/
    struct edgeDescription 
    {
        int id;    
        uint64_t memoryRequirement;
        std::string name;
        uint64_t flow;
        int length;

        edgeDescription(int id=0, uint64_t m = 0, std::string aname = "", uint64_t f = 0, int l = 1) : 
            id(id), memoryRequirement(m), name(aname), flow(f), length(l) { }
    };

    class LemonGraphScheduler : public LogSender
    {
        lemon::ListDigraph graph_;

        lemon::ListDigraph::NodeMap<nodeDescription> nodes_;
        lemon::ListDigraph::ArcMap<edgeDescription> edges_;
        lemon::ListDigraph::ArcMap<uint64_t> edgesMemory_;
        lemon::ListDigraph::ArcMap<int> edgesLength_;

        lemon::ListDigraph::Node graphSourceNode_;
        lemon::ListDigraph::Node graphSinkNode_;

        /*New edges added to the graph from partial serialisation, these will be added to the McM graph*/
        std::vector<mv::edgeDescription> partialSerialisationEdgesAdded_;
           
    public:
        LemonGraphScheduler();
        ~LemonGraphScheduler();
        lemon::ListDigraph& getGraph();

        void convertMcMGraphToLemonGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
        
        bool performPartialSerialisation(const mv::pass::PassEntry& pass, std::vector<mv::edgeDescription> cutEdges, mv::ComputationModel& model);
        std::pair<int, std::vector<edgeDescription>> calculateMaxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
        uint64_t calculateFMax(mv::ComputationModel& model);
        void insertpartialSerialisationEdgesInMcmGraph(mv::ComputationModel& model);
          
        std::string getLogID() const override;
    };
}

#endif 
