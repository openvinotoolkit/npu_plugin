#ifndef MV_LEMONGRAPHSCHEDULER
#define MV_LEMONGRAPHSCHEDULER

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"

#include <lemon/list_graph.h>

// #include "../../../../contrib/koala/graph/graph.h"
// #include "../../../../contrib/koala/algorithm/conflow.h"
// #include "../../../../contrib/koala/algorithm/weights.h"
// #include "../../../../contrib/koala/io/graphml.h"
// #include "../../../../contrib/koala/io/text.h"
// #include "../../../../contrib/koala/io/parsetgml.h"
// #include "../../../../contrib/koala/classes/create.h"


namespace mv
{
    class LemonGraphScheduler : public LogSender
    {
        lemon::ListDigraph* graph_;

        /*KOALA vertices and edges iterators*/
        //std::vector<koalaGraph::PVertex> vertices_;  
        //std::vector<koalaGraph::PEdge> edges_;
        
        /*New edges added to the graph from partial serialisation, these will be added to the McM graph*/
        //std::vector<koalaGraph::PEdge> partialSerialisationEdgesAdded_;

        //std::vector<koalaGraph::PEdge>::const_iterator lookUpKoalaEdgebyName(std::string edgeName, const std::vector<koalaGraph::PEdge>& koalaEdges);
        //std::vector<mv::koalaGraph::PVertex>::const_iterator lookUpKoalaSinkNode(bool sinknode, const std::vector<koalaGraph::PVertex>& koalaVertices);
        //std::vector<mv::koalaGraph::PVertex>::const_iterator lookUpKoalaSourceNode(bool sourcenode, const std::vector<koalaGraph::PVertex>& koalaVertices);
           
    public:
        LemonGraphScheduler();
        ~LemonGraphScheduler();
        lemon::ListDigraph& getGraph();

        //void convertMcMGraphToLemonGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
        
        //void performPartialSerialisation(const mv::pass::PassEntry& pass, std::vector<koalaGraph::PEdge> cutEdges);
        //std::pair<int,std::vector<koalaGraph::PEdge>> calculateMaxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
        //uint64_t calculateFMax(mv::ComputationModel& model);
        //void insertpartialSerialisationEdgesInMcmGraph(mv::ComputationModel& model);
          
        //std::string getLogID() const override;

    };
}

#endif 
