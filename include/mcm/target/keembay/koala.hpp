#ifndef MV_KOALA
#define MV_KOALA

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "../../../../contrib/koala/graph/graph.h"
#include "../../../../contrib/koala/algorithm/conflow.h"
#include "../../../../contrib/koala/algorithm/weights.h"
#include "../../../../contrib/koala/io/graphml.h"
#include "../../../../contrib/koala/io/text.h"
#include "../../../../contrib/koala/io/parsetgml.h"
#include "../../../../contrib/koala/classes/create.h"


namespace mv
{
    /*KOALA Node Description*/
    struct nodeDescription {

	std::string name;
    int cost; /* Required for optimal partial serialisation (adding edge that minimises increase in the critical path)*/
    bool sourceNode;
    bool sinkNode;
    
    nodeDescription(std::string aname = "", int cost = 0, bool sourcenode = false, bool sinknode = false) :name(aname), cost(cost), sourceNode(sourcenode), sinkNode(sinknode){}  
    };

    /*KOALA Edge Description*/
    struct edgeDescription {

	int memoryRequirement;
	std::string name;
    int flow;
    int length;

    edgeDescription(int m = 0, std::string aname = "", int f = 0, int l = 1) : memoryRequirement(m), name(aname), flow(f), length(l) {}
    };

    /*Define KOALA graph's node and edge type*/
    using koalaGraph = Koala::Graph <nodeDescription, edgeDescription>;
    
    class Koala : public LogSender 
    {
        /*Vectors to store KOALA vertices and edges iterators*/
        std::vector<koalaGraph::PVertex> vertices; 
        std::vector<koalaGraph::PEdge> edges;
    
    public:
        // Koala(const std::string& name);
        koalaGraph::PVertex lookUpKoalaVertexbyName(const std::string& vertexName, const std::vector<koalaGraph::PVertex>& koalaVertices);
        koalaGraph::PVertex lookUpKoalaSinkNode(bool sinknode, const std::vector<koalaGraph::PVertex>& koalaVertices);
        koalaGraph::PVertex lookUpKoalaSourceNode(bool sourcenode, const std::vector<koalaGraph::PVertex>& koalaVertices);
        koalaGraph::PEdge lookUpKoalaEdgebyName(std::string edgeName, const std::vector<koalaGraph::PEdge>& koalaEdges);
        void convertMcMGraphToKoalaGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model, koalaGraph& flowGraph, std::vector<koalaGraph::PVertex>& vertices, std::vector<koalaGraph::PEdge>& edges);
        int calculateFMax(mv::ComputationModel& model);
        void insertpartialSerialisationEdgesInMcmGraph(mv::ComputationModel& model, std::vector<koalaGraph::PEdge>& partialSerialisationEdgesAdded)
        int performPartialSerialisation(const mv::pass::PassEntry& pass, koalaGraph& flowGraph, int cutValue, std::vector<koalaGraph::PEdge> cutEdges, 
                                koalaGraph::PVertex graphSource, koalaGraph::PVertex graphSink, std::vector<koalaGraph::PVertex>& vertices, 
                                std::vector<koalaGraph::PEdge>& edges, std::vector<koalaGraph::PEdge>& partialSerialisationEdgesAdded);
        std::pair<int,std::vector<koalaGraph::PEdge>> calculateMaxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, koalaGraph& flowGraph, std::vector<koalaGraph::PVertex>& Vertices, std::vector<koalaGraph::PEdge>& Edges);
    };
}

#endif 
