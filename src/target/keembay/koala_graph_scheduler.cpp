#include "include/mcm/target/keembay/koala_graph_scheduler.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::KoalaGraphScheduler::KoalaGraphScheduler(): graph_(new koalaGraph)
{
    
}

mv::KoalaGraphScheduler::~KoalaGraphScheduler()
{
    delete graph_;
}

mv::koalaGraph& mv::KoalaGraphScheduler::getGraph()
{
    return *graph_;
}

/**
 * @brief Returns a KOALA vertex iterator corresonding to the sink node of the KOALA graph 
 * @param sinkNode - attribute of the KOALA node indicating if it is the sink node (true) 
 * @param koalaVertices - vector of KOALA vertices iterators
 * @return An iterator to a KOALA vertex iterator 
 */

std::vector<mv::koalaGraph::PVertex>::const_iterator mv::KoalaGraphScheduler::lookUpKoalaSinkNode(bool sinknode, const std::vector<koalaGraph::PVertex>& koalaVertices) {
 
    for(auto iter = koalaVertices.begin(); iter != koalaVertices.end(); ++iter) {
        if((*iter)->info.sinkNode == sinknode) 
            return iter; 
    }
    throw std::runtime_error("Could not find Koala graph sink node, exit");
}

/**
 * @brief Returns a KOALA vertex iterator corresonding to the source node of the KOALA graph 
 * @param sinkNode - attribute of the KOALA node indicating if it is the source node (true) 
 * @param koalaVertices - vector of KOALA vertices iterators
 * @return An iterator to a KOALA vertex iterator
 */

std::vector<mv::koalaGraph::PVertex>::const_iterator mv::KoalaGraphScheduler::lookUpKoalaSourceNode(bool sourcenode, const std::vector<koalaGraph::PVertex>& koalaVertices) {
    
    for(auto iter = koalaVertices.begin(); iter != koalaVertices.end(); ++iter) {
        if((*iter)->info.sourceNode == sourcenode) 
            return iter;
    }
    throw std::runtime_error("Could not find Koala graph source node, exit"); 
}

/**
 * @brief Returns a KOALA edge iterator corresonding to the name of the iterator 
 * @param edgeName - the name of the KOALA vertex you are searching for
 * @param koalaEdges - vector of KOALA edges iterators
 * @return An iterator to a KOALA vertex iterator
 */

std::vector<mv::koalaGraph::PEdge>::const_iterator mv::KoalaGraphScheduler::lookUpKoalaEdgebyName(std::string edgeName, const std::vector<koalaGraph::PEdge>& koalaEdges) {
    
    for(auto iter = koalaEdges.begin(); iter != koalaEdges.end(); ++iter) {
        if((*iter)->info.name == edgeName) 
            return iter;
    }
    throw std::runtime_error("Could not find edge by name in the Koala graph, exit");
}
/**
 * @brief Convert McM graph (control model view) to KOALA graph and store the data required to perform the max topoloigcal cut algorithm on the KOALA graph edges
 * @param pass  - pass object
 * @param model - McM computation model
 */
void  mv::KoalaGraphScheduler::convertMcMGraphToKoalaGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {

    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    /* For each task in the ControlModel view of the MCM graph
     * create a corresponding node (task) in the KOALA graph.
     * Add all the nodes to the KOALA graph first and then add the edges.
    */
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt)
    {

       /* We do not require MCM constant operations and MCM ouput operation in the KOALA graph. The sink node in the KOALA graph is the DMATask CMX2DDR.
        * For all other tasks in the ControlModel view of the MCM graph create a corresponding node in the KOALA graph.
       */
       if (opIt->getOpType() != "ConstantDataElement" && opIt->getOpType() != "Output" && opIt->getOpType() != "ConstantInt" &&
            opIt->getOpType() != "WeightsTable" && opIt->getOpType() != "SparsityMap") {

           bool nodeAdded = false;
           /*Add node to KOALA graph*/
           /*Check if the node is a DMA task CMX to DDR (this is the sink node in KOALA graph and we need to keep track of it)*/
           if ((opIt->getOpType() == "DMATask") && (opIt->get<mv::DmaDirection>("direction") == mv::CMX2DDR)) {

               pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());

               this->vertices_.push_back(this->getGraph().addVert(nodeDescription(opIt->getName(),0, false, true)));
               nodeAdded = true;
           }
           
           /*Keep track of the source node i.e. input*/
           if (opIt->getOpType() == "Input") { 
               pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());
               this->vertices_.push_back(this->getGraph().addVert(nodeDescription(opIt->getName(),0, true, false)));
               nodeAdded = true;
           }
           /*All other nodes between source and sink node*/
           else if (!nodeAdded) {
               
               pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());

               this->vertices_.push_back(this->getGraph().addVert(nodeDescription(opIt->getName(),0, false,false)));
               nodeAdded = true;
           }
       }
    }
    
    /* Add the edges to the KOALA graph store attributes on the edges to perform the max topoloigcal cut algorithm.
     * Iterate over the the control flow edges in the MCMgraph.  
    */
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {
        
        /* 1. Don't add the edge going to Ouput in the MCM graph to the KOALA graph
         * 2. Don't add edge coming from a ConstantInt operation (Sparsity Map and Weights Table)
        */

        if (flowIt.sink()->getOpType() != "Output" && flowIt.source()->getOpType() != "ConstantInt" &&
            flowIt.source()->getOpType() != "WeightsTable" && flowIt.source()->getOpType() != "SparsityMap") {

            auto sourceName = flowIt.source()->getName();
            auto sinkName  = flowIt.sink()->getName();

            if(flowIt->hasAttr("MemoryRequirement"))
                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + sourceName + " --> " + sinkName + " with memory requirement " + std::to_string(flowIt->get<int>("MemoryRequirement")));
            else
                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + sourceName + " --> " + sinkName + " with memory requirement " + std::to_string(0));

            /*If the control flow has a memoryRequirment attribute add it to the KOALA edge*/
            if(flowIt->hasAttr("MemoryRequirement"))
                this->edges_.push_back(this->getGraph().addEdge(*std::find_if(vertices_.begin(), vertices_.end(), [&sourceName](koalaGraph::PVertex const& vertex) {return sourceName == vertex->info.name;}), 
                                             *std::find_if(vertices_.begin(), vertices_.end(), [&sinkName](koalaGraph::PVertex const& vertice) {return sinkName == vertice->info.name;}), 
                                             edgeDescription(flowIt->get<int>("MemoryRequirement"),flowIt->getName()), 
                                             Koala::Directed));
                                             
            /*Otherwsise the memory requirment is 0*/
            else
                this->edges_.push_back(this->getGraph().addEdge(*std::find_if(vertices_.begin(), vertices_.end(), [&sourceName](koalaGraph::PVertex const& vertex) {return sourceName == vertex->info.name;}), 
                                             *std::find_if(vertices_.begin(), vertices_.end(), [&sinkName](koalaGraph::PVertex const& vertex) {return sinkName == vertex->info.name;}), 
                                             edgeDescription(0,flowIt->getName()), 
                                             Koala::Directed));
        }
    }
    std::cout << std::to_string(this->getGraph().getVertNo()) << std::to_string(this->getGraph().getEdgeNo()) << std::endl;
    pass.log(mv::Logger::MessageType::Debug, "KOALA graph has " + std::to_string(this->getGraph().getVertNo()) + " vertices and " + std::to_string(this->getGraph().getEdgeNo()) + " edges");
}

uint64_t mv::KoalaGraphScheduler::calculateFMax(mv::ComputationModel& model) {

    mv::ControlModel cm(model);

    /*Compute Fmax - (defined as sum of memory requirments + 1)*/
    uint64_t Fmax = 0;
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {
        if(flowIt->hasAttr("MemoryRequirement")) {
            Fmax += flowIt->get<int>("MemoryRequirement");
        }
    }
    Fmax += 1; /*add 1 to Fmax as per algorithm*/
    return Fmax;
}

void mv::KoalaGraphScheduler::insertpartialSerialisationEdgesInMcmGraph(mv::ComputationModel& model) {

    std::set<std::pair<std::string, std::string>> addedEdges;
    for (const auto& edge : partialSerialisationEdgesAdded_) {
        
        std::string edgeSourceName = edge->getEnd1()->info.name;
        std::string edgeSinkName = edge->getEnd2()->info.name;

        mv::ControlModel cm(model);

        mv::Control::OpListIterator mcmSourceNodeIterator;
        mv::Control::OpListIterator mcmSinkNodeIterator;

        /*Find the McM iterator for the source node*/
        for (auto opItSource = cm.getFirst(); opItSource != cm.opEnd(); ++opItSource) {
            
            if(opItSource->getName() == edgeSourceName) 
                mcmSourceNodeIterator = opItSource;
        }

        /*Find the McM iterator for the sink node*/
        for (auto opItSink = cm.getFirst(); opItSink != cm.opEnd(); ++opItSink) {
            
            if(opItSink->getName() == edgeSinkName) 
                mcmSinkNodeIterator = opItSink;
        }
        auto inserted = addedEdges.insert(std::make_pair(edgeSourceName, edgeSinkName));
        if (inserted.second)
        {
            /*Add the edge to graph*/
            auto partialSerialisationEdge = cm.defineFlow(mcmSourceNodeIterator, mcmSinkNodeIterator);
            partialSerialisationEdge->set<bool>("PartialSerialisationEdge", true);
        }
    }
}

/**
 * @brief Perform partial serilisation of KOALA graph to reduce maximum peak memory
 * @param cutValue  - Maximum peak memory of the graph
 * @param cutEdges - Vector of cut edges from the max topological cut
 * @param graphSource - Source node of KOALA graph
 * @param graphSink - Sink node of KOALA graph
 * @param vertices - Vector of KOALA vertices iterators
 * @param edges - Vector of KOALA edge iterators
 * @return - 0 success
 */
void mv::KoalaGraphScheduler::performPartialSerialisation(const mv::pass::PassEntry& pass, std::vector<koalaGraph::PEdge> cutEdges) {
    
    /* Partial serialisation works by getting the source and sink nodes of the cutEdges returned from max topoloigcal cut
     * It then creates a pool of all possible edges that it can add to the graph using these source and sink nodes.
     * Do not include the original cut edges in this pool as they are already in the graph.
     * The direction of the new edge is however in the opposite direction, sink --> source. Take care to ensure the correct direction. 
    */

    /*Sources and sinks of cut edges*/
    std::vector<koalaGraph::PVertex> sources;
    std::vector<koalaGraph::PVertex> sinks;

    /*Pool of possible edges to add to the graph expressed as souce and sink nodes*/
    std::vector<std::pair<koalaGraph::PVertex,koalaGraph::PVertex>> possibleEdges;

    /*Cut edges source and sink vectors*/
    std::vector<std::pair<koalaGraph::PVertex,koalaGraph::PVertex>> cutEdgesSourceSink;

    /*Get the source and sink of each cut edge*/
    for (const auto& edge : cutEdges)
        cutEdgesSourceSink.push_back(std::make_pair(edge->getEnd1(), edge->getEnd2()));
    
    /*Get cut edges sources*/
    for (const auto& edge : cutEdges) {
        if(std::find(sources.begin(), sources.end(), edge->getEnd1()) != sources.end()) {   
            /* sources already contains the edge source node */
        } else {
            /* add edge source node to sources */
            sources.push_back(edge->getEnd1());
        }   
    }

    /*Get cut edges sinks*/
    for (const auto& edge : cutEdges) {
        if(std::find(sinks.begin(), sinks.end(), edge->getEnd2()) != sinks.end()) {   
            /* sources already contains the edge sink node */
        } else {
            /* add edge sink node to sources */
            sinks.push_back(edge->getEnd2());
        }   
    }

    /*Create pool of possible partial serialisation edges but not including original cutset edges*/
    for (const auto& sinknode : sinks) {
        for (const auto& sourcenode : sources) {
            bool found = false;

            for(int i = 0; i < cutEdgesSourceSink.size(); i++) {
                
                /*Check if new potential partial serialisation edge is an original cut set edge, if it is then do not add it to the pool*/
                if((cutEdgesSourceSink[i].first->info.name == sourcenode->info.name) && (cutEdgesSourceSink[i].second->info.name == sinknode->info.name)) {
                     found = true;
                     break;
                }
    
            }
            if (!found) /*Edge not found in original cut set therefore add it to the pool*/
                possibleEdges.push_back(std::make_pair(sourcenode, sinknode));
        }
    }
    
    /* Attempt to add each edge to edge in the graph and check if it is still a DAG*/
    /* It is still a DAG then recalculate max topological cut*/
    /* Note in future here is where the optimal edge should be selected such that it minimises the increase in the critical path of the graph*/

  
    for(std::size_t i = 0; i < possibleEdges.size(); i++) {

        auto sourceName = possibleEdges[i].second->info.name;
        auto sinkName  = possibleEdges[i].first->info.name;

        pass.log(mv::Logger::MessageType::Debug, "Adding partial serialisation edge to KOALA graph from: " + sourceName + " --> " + sinkName );

        auto newEdge = this->getGraph().addEdge(*std::find_if(vertices_.begin(), vertices_.end(), [&sourceName](koalaGraph::PVertex const& vertex) {return sourceName == vertex->info.name;}), 
                                             *std::find_if(vertices_.begin(), vertices_.end(), [&sinkName](koalaGraph::PVertex const& vertex) {return sinkName == vertex->info.name;}), 
                                             edgeDescription(0,"PS_edge_"+sinkName+sourceName), 
                                             Koala::Directed);
        /*get number of vertices*/
        int n = this->getGraph().getVertNo();
	    koalaGraph::PVertex LOCALARRAY(tabV, n);
		
        /* Get topological order*/
	    Koala::DAGAlgs::topOrd(this->getGraph(), tabV); 
		
        /*Check if it is a DAG*/
	    bool isDag = Koala::DAGAlgs::isDAG(this->getGraph(), tabV, tabV + n);

        if(isDag) {
            pass.log(mv::Logger::MessageType::Debug, "The graph is still a DAG after adding partial serialisation edge, recalulating max topological cut value");
            
            /*add edge iterator to the vector of KOALA edge iterators*/
            edges_.push_back(newEdge); 

            /*keep track of the edges added as these edges will be added to mcmGraph*/
            partialSerialisationEdgesAdded_.push_back(newEdge);
            return;
        }
        else {
            pass.log(mv::Logger::MessageType::Debug, "Removing partial serialisation edge as graph is no longer a DAG, from: " + sourceName + " --> " + sinkName );
            this->getGraph().delEdge(newEdge);
        }
    }
    throw std::runtime_error("The maximum peak memory requirment of the graph exceeds CMX and the partial serialisation algorithm is unable to reduce parallelism, exiting now, this is normal behaviour");
}

/*
 * See Max topological cut algorithm description in this paper:
 * 
 * L. Marchal, H. Nagy, B. Simon and F. Vivien, "Parallel Scheduling of DAGs under Memory Constraints," 
 * 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS), Vancouver, BC, 2018, pp. 204-213.
 * doi: 10.1109/IPDPS.2018.00030 
*/ 
std::pair<int,std::vector<mv::koalaGraph::PEdge>> mv::KoalaGraphScheduler::calculateMaxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {

    mv::ControlModel cm(model);

    /* Calculate Fmax - Defined as sum of memory requirments + 1)*/
    auto Fmax = this->calculateFMax(model); 
    
    /*Perform the max topological cut algorithm here*/

    /*See the shortest path KOALA example here: http://koala.os.niwa.gda.pl/api/examples/weights/dijkstra_h/dijkstra_h.html*/

    Koala::AssocArray <koalaGraph::PEdge, Koala::DijkstraHeap::EdgeLabs<int>> edgeMap; /*input container*/
    Koala::AssocArray <koalaGraph::PVertex, Koala::DijkstraHeap::VertLabs<int,koalaGraph>> vertMap; /*output container*/
     
    /* Construct the graph demand: cicle over the edge and add
     * a flow equal to Fmax on a shorest path containing that node
    */

    /*containter to store the edges on shorest paths*/
    std::vector <koalaGraph::PEdge> shortestPathEdges;
    koalaGraph::PEdge LOCALARRAY(edges, this->getGraph().getEdgeNo());
    int numberofEdges = this->getGraph().getEdges(edges);
 
    /*For each edge
     *
     * Find the shortest path from source node (Input) to the edges source node and
     * Find the shortest path from the edges sink node to the sink node (DMA task CMX to DDR) 
    */
    for (int i = 0; i < numberofEdges; i++) {

        /*get the source and sink node of the edge*/
        pass.log(mv::Logger::MessageType::Debug, "Source Node " + this->getGraph().getEdgeEnds(this->edges_[i]).first->info.name);
        pass.log(mv::Logger::MessageType::Debug, "Sink Node " + this->getGraph().getEdgeEnds(this->edges_[i]).second->info.name);

        /*Find the shortest path from the input node to the source node of the edge*/
        Koala::DijkstraHeap::PathLengths <int> resInputToSource = Koala::DijkstraHeap::findPath(this->getGraph(), 
                                                                                                edgeMap, 
                                                                                                (*lookUpKoalaSourceNode(true, this->vertices_)),
                                                                                                this->getGraph().getEdgeEnds(this->edges_[i]).first, 
                                                                                                Koala::DijkstraHeap::outPath(blackHole, back_inserter(shortestPathEdges)));

        pass.log(mv::Logger::MessageType::Debug, "Number of edges on the path from Input to source node of the current edge is " + std::to_string(resInputToSource.edgeNo));

	    for (int k = 0; k < resInputToSource.edgeNo; k++) {

            pass.log(mv::Logger::MessageType::Debug, shortestPathEdges[k]->info.name);

            /*Add Fmax to the flow attribute of the edge*/
            auto edge = lookUpKoalaEdgebyName(shortestPathEdges[k]->info.name, this->edges_);
            (*edge)->info.flow +=Fmax;
	    }

        /*The above calculation stops at source node of the edge so doesn't include the edge in question - add Fmax to this edge*/
        this->edges_[i]->info.flow +=Fmax;

        /*Clear the container used to store the the edges on shorest paths*/
        shortestPathEdges.clear(); 

        /*Find the shortest path from the sink node of the edge to the sink node (DMA task CMX to DDR)*/
        Koala::DijkstraHeap::PathLengths <int> resSinkToOuput = Koala::DijkstraHeap::findPath(this->getGraph(), 
                                                                                                edgeMap, 
                                                                                                this->getGraph().getEdgeEnds(this->edges_[i]).second, 
                                                                                                (*lookUpKoalaSinkNode(true, this->vertices_)), 
                                                                                                Koala::DijkstraHeap::outPath(blackHole, back_inserter(shortestPathEdges)));

        pass.log(mv::Logger::MessageType::Debug, "Number of edges on the path from the sink node of the current edge to the ouput node is " + std::to_string(resSinkToOuput.edgeNo));

	    for (int j = 0; j < resSinkToOuput.edgeNo; j++) {
		    
            pass.log(mv::Logger::MessageType::Debug, shortestPathEdges[j]->info.name);

            /*Add Fmax to the flow attribute of the edge*/
            auto edge = lookUpKoalaEdgebyName(shortestPathEdges[j]->info.name, this->edges_);
            (*edge)->info.flow +=Fmax;
	    }
        /*Clear the container used to store the the edges on shorest paths*/
        shortestPathEdges.clear();
    }

    /*Subtract Memory attribute of edge from the Flow attribute of the edge*/
    for (int i = 0; i < numberofEdges; i++)
		this->edges_[i]->info.flow = this->edges_[i]->info.flow - this->edges_[i]->info.memoryRequirement;
    

    /* Perform Min cut on the graph, see this example: http://koala.os.niwa.gda.pl/api/examples/flow/example_Flow.html*/
    /* Set edge capacities (flow attribute of the edge ) and costs (=1)*/
	Koala::AssocArray< koalaGraph::PEdge, Koala::Flow::EdgeLabs<uint64_t,int>> cap;

    for (int i = 0; i < numberofEdges; i++) {
        cap[this->edges_[i]].capac = this->edges_[i]->info.flow; 
        cap[this->edges_[i]].cost = 1;
    }

    /*store the cut edges*/
    std::vector<koalaGraph::PEdge> cutEdges;
    uint64_t maxTopologicalCutValue = 0;

    /*compute minimal cut*/
    Koala::Flow::minEdgeCut(this->getGraph(), cap, (*lookUpKoalaSourceNode(true, this->vertices_)), (*lookUpKoalaSinkNode(true, this->vertices_)), Koala::Flow::outCut(blackHole, std::back_inserter(cutEdges)));
    
    for (std::size_t i = 0; i < cutEdges.size(); i++)
        maxTopologicalCutValue += cutEdges[i]->info.memoryRequirement;

    /*Add Max topological cut value as attribute to output node*/
    auto output = cm.getOutput();
    output->set<uint64_t>("MaxTopologicalCutValue", maxTopologicalCutValue); 

    pass.log(mv::Logger::MessageType::Debug, "The maximum peak memory of the graph is " + std::to_string(maxTopologicalCutValue) + " bytes");

    return std::make_pair(maxTopologicalCutValue, cutEdges);
}

std::string mv::KoalaGraphScheduler::getLogID() const
{
    return "Koala";
}



