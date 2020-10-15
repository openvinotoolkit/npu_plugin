#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/algorithms/dijkstra.hpp"

namespace mv {
namespace graphOptimizer  {

using namespace std;

bool MetaEdge::operator==(const MetaEdge& other)
{
    return id == other.id ;
}

bool MetaEdge::operator<(const MetaEdge& other)
{
    return id < other.id;
}

bool MetaEdge::operator>(const MetaEdge& other)
{
    return id > other.id;
}

bool MetaEdge::operator!=(const MetaEdge& other)
{
    return id != other.id;
}

double MetaEdge::operator+(const double other)
{
    return cost_ + other;
}

double MetaEdge::operator+(const MetaEdge& other)
{
    return cost_ + other.cost_;
}

MetaEdge& MetaEdge::operator+=(const double other)
{
    this->cost_ += other;
    return *this;
}

MetaEdge& MetaEdge::operator+=(const MetaEdge& other)
{
    this->cost_ += other.cost_;
    return *this;
}

void MetaEdge::extend(const CriticalPathNodes& childCriticalPath)
{
    criticalPath_.reserve(criticalPath_.size() + distance(childCriticalPath.begin(),childCriticalPath.end()));
    criticalPath_.insert(criticalPath_.end(),childCriticalPath.begin(),childCriticalPath.end());
}

const MetaEdge::CriticalPathNodes& MetaEdge::criticalPaths() const
{
    return criticalPath_;
}
double MetaEdge::cost() const
{
    return cost_;
}

void MetaGraph::StrategySetPair::operator=(const StrategySetPair& other)
{
    parent = other.parent;
    child = other.child;
    return;
}

void MetaGraph::StrategySetPair::print() const
{
    cout<<  (*parent)["name"].get<string>() <<
            " id " << (*parent)["id"].get<int>() <<
            " ==> " << (*child)["name"].get<string>() <<
            " id " << (*child)["id"].get<int>();
    return;
}

size_t MetaGraph::StrategySetHash::operator ()(const StrategySetPair& val) const
{
    //since this is a hash operator, the parameter must be const. we cannot use
    //firstId = val.first["id"].get<int>() since the operator[] of the map is not const....
    auto parentId = val.parent->find("id");
    auto childId = val.child->find("id");

    auto parentIdVal = parentId->second.get<int>();
    auto childIdVal = childId->second.get<int>();

    //todo:: unmagify the number 100
    int hash = parentIdVal * 100 + childIdVal;

    return (hash);
}

bool MetaGraph::StrategytSetCompare::operator()(const StrategySetPair& lhs,
                const StrategySetPair& rhs) const
{
    auto leftFirstId  = lhs.parent->find("id")->second.get<int>();
    auto rightFirstId = rhs.parent->find("id")->second.get<int>();

    auto leftSecondId  = lhs.child->find("id")->second.get<int>();
    auto rightSecondId = rhs.child->find("id")->second.get<int>();

    return ((leftFirstId == rightFirstId) and (leftSecondId == rightSecondId));
}

bool MetaGraph::costEdgeIteratorComp::operator()(const OptimizationGraph::edge_list_iterator lhs,
                                                    const OptimizationGraph::edge_list_iterator rhs) const
{
    return (*lhs) < (*rhs);
}

bool MetaGraph::costNodeIteratorComp::operator()(const OptimizationGraph::node_list_iterator lhs,
                                                    const OptimizationGraph::node_list_iterator rhs) const
{
    int x = (*lhs)["id"];
    int y = (*rhs)["id"];
    return (x < y) ;
}

void MetaGraph::addNewLevel(Op& op,
                            shared_ptr<vector<StrategySet>> newLevel,
                            function<double(Op&,Op&,StrategySet&,StrategySet&)> cost)
{
    levelContainer_.push_back(newLevel);
    if( levels.size() == 0)
    {
        //todo::do we need to copy???
        auto& newSet = *(levelContainer_.back());

        //todo: we know the size of the level...
        levels.push_back(GraphLevel(&op));
        auto& latestLevel = levels.back();

        for(unsigned strategyCtr = 0 ; strategyCtr < newSet.size(); ++ strategyCtr)
        {
            //todo::don't assume map over here
            //todo:this makes a pretty dangerous assumption that the strategySet vector will not be changed,
            // and the elems inside do not move!!!!!
            latestLevel.level.push_back(internalGraph_.node_insert(newSet[strategyCtr]));
        }
        firstLevelIdx_ = 0;
        lastLevelIdx_ = 0;

    }
    else
    {
        //todo::do we need to copy???
        levels.push_back(GraphLevel(&op));

        auto& newSet = *(levelContainer_.back());
        auto& latestLevel = levels.back();

        auto& lastLevel = levels[lastLevelIdx_];

        for(unsigned strategyCtr = 0 ; strategyCtr < newSet.size(); ++ strategyCtr)
        {
            //todo:this makes a pretty dangerous assumption that the strategySet vector will not be changed,
            // and the elems inside do not move!!!!!
            latestLevel.level.push_back(internalGraph_.node_insert(newSet[strategyCtr]));
        }

        for(const auto oldNode : lastLevel.level){
            for(const auto newNode : latestLevel.level)
            {
                double edgeCost = cost(*lastLevel.op,*latestLevel.op, *oldNode, *newNode);
                if (edgeCost == numeric_limits<double>::infinity())
                    continue;

                auto newEdge = internalGraph_.edge_insert(oldNode,newNode,MetaEdge(edgeCost));

                edgeCostMap.insert(pair<OptimizationGraph::edge_list_iterator, double>(newEdge, edgeCost));
            }
        }
        lastLevelIdx_ = levels.size() - 1;
    }
}

void MetaGraph::fuseMeta(shared_ptr<MetaGraph> childGraph)
{
    if(childGraph->solved_ == false)
    {
        //todo:: throw exception. Cannot fuse an unsolved Meta
    }

    childMetaGraphs.push_back(childGraph);

    auto& childFirstLevel = childGraph->levels[childGraph->firstLevelIdx_];
    auto& childLastLevel  = childGraph->levels[childGraph->lastLevelIdx_];

    //when fusing a child meta, we will make the assumption that it has a set of firstLevel and lastLevel nodes.
    //if the childMeta is solved, then we have a criticalPath, between all combinations of firstLevel and lastLevels.
    //the nodes of the parent meta, will be the firstLevel and lastLevel nodes of it's composing children metas
    //the edges of the parent meta, will be the criticalPaths corresponding to the firstNodes and lastNodes.

    if( levels.size() == 0)
    {
        //if we are the first childMeta to be fused, then set the firstLevel and lastLevel of the childMeta.
        levels.push_back(GraphLevel(childFirstLevel.op));
        levels.push_back(GraphLevel(childLastLevel.op));

        auto& firstLevel = levels[0];
        auto& lastLevel = levels[1];

        for(unsigned strategyCtr = 0 ; strategyCtr < childFirstLevel.level.size(); ++ strategyCtr)
            firstLevel.level.push_back(internalGraph_.node_insert(*childFirstLevel.level[strategyCtr]));
        for(unsigned strategyCtr = 0 ; strategyCtr < childLastLevel.level.size(); ++ strategyCtr)
            lastLevel.level.push_back(internalGraph_.node_insert(*childLastLevel.level[strategyCtr]));

        //for the new pair levels, add the edges of this graph. iterate over all the combinations, and
        //find the criticalPath for that respective pair, and make and edge out of it
        for(auto& parent : firstLevel.level)
            for(auto& child : lastLevel.level)
            {
                //todo:: cosnider strategySetPair to take iterators? or consider std::pair?
                const StrategySetPair path(&(*parent),&(*child));
                auto elem = childGraph->criticalPaths_.find(path);

                if(elem == childGraph->criticalPaths_.end())
                {
//                    todo:: raise exception
//                        cout<<"ERROR ERROR COULD NOT FIND CRIPATH IN FUSION" << endl;
//                        cout<<"searchingFor p "; path.print(); cout<< endl;
//                        for(auto m : childGraph.criticalPaths_)
//                        {
//                            cout<<"Have " ;m.first.print(); cout << "with cost " << m.second.cost <<  endl;
//                        }
                }
//                    cout<<"Found " ; path.print(); cout << "with cost " << elem->second.cost <<  endl;
                //todo: keep criNodes

                auto& criPath = elem->second;
                auto newEdge = internalGraph_.edge_insert(parent,child,MetaEdge(criPath.cost));
                (*newEdge).extend(*criPath.nodes);

                edgeCostMap.insert(pair<OptimizationGraph::edge_list_iterator, double>(newEdge, criPath.cost));
            }
        firstLevelIdx_ = 0;
        lastLevelIdx_ = 1;

        return;
    }
    else
    {
        //need to search the firstLevel and lastLevel sets from the childGraph, inside the levels of the graph of
        //the parent
        int parentLevelCtr = -1;
        int childLevelCtr = -1;
        //search for our pivots in the master graph
        for(unsigned levelCtr = 0; levelCtr < levels.size() ; ++levelCtr)
        {
            if( (*(levels[levelCtr].op)) == (*childFirstLevel.op))
                parentLevelCtr = levelCtr;
            if( (*(levels[levelCtr].op)) == (*childLastLevel.op))
                childLevelCtr = levelCtr;

            if( (parentLevelCtr != -1) and (childLevelCtr != -1))
                break;
        }

        if( (parentLevelCtr != -1) and (childLevelCtr != -1))
        {
            //both levels are existing. then we take each edge between the levels in the parent graph, and
            //extend the edge info, by increasing the cost of the edge with the childGraph's critical path
            //and extend the container of the criticalPathNodes (aka the strategies)

            //todo::for now we assume that the "children" of the parent in the master graph coresponds to the
            //      lastLevel inside the childGraph. todo: actully do a check for this. Should not happen, but need to check
            auto& parentLevel = levels[parentLevelCtr];
            for(const auto& parent : parentLevel.level)
                for(auto edge = parent->leftmost_output(); edge != internalGraph_.edge_end(); ++edge )
                {
                    auto child = edge->sink();

                    //todo:: cosnider strategySetPair to take iterators? or consider std::pair?
                    const StrategySetPair path(&(*parent),&(*child));
                    auto elem = childGraph->criticalPaths_.find(path);

                    if(elem == childGraph->criticalPaths_.end())
                    {
                        //todo:: raise exception
//                        cout<<"ERROR ERROR COULD NOT FIND CRIPATH IN FUSION" << endl;
//                        cout<<"searchingFor p " << path.print(); cout<< endl;
//                        for(auto m : childGraph.criticalPaths_)
//                            cout<<"Have " << m.first.print(); cout << endl;
                    }
//                    cout<<"Found " ; path.print(); cout << endl;

                    auto& criPath = elem->second;
                    const auto cost = criPath.cost;

                    //todo::extend with cripath;
                    (*edge).operator +=(cost);
                    (*edge).extend(*criPath.nodes);

                    edgeCostMap[edge] = (*edge).cost();
                }

            //in this case we will not update the first/last level indexes in the graph, since this was not an
            //addition but a fusion
        }
        else if( (parentLevelCtr != -1) and (childLevelCtr == -1))
        {
            //if we found only the parent, then we will append a new set of nodes, aka, the children.

            levels.push_back(GraphLevel(childLastLevel.op));
            auto& lastLevel = levels.back();
            auto& parentLevel = levels[parentLevelCtr];

            for(unsigned strategyCtr = 0 ; strategyCtr < childLastLevel.level.size(); ++ strategyCtr)
            {
                lastLevel.level.push_back(internalGraph_.node_insert(*childLastLevel.level[strategyCtr]));
            }

            for(const auto& parent : parentLevel.level)
                for(const auto& child : lastLevel.level)
                {
                    const StrategySetPair path(&(*parent),&(*child));
                    auto elem = childGraph->criticalPaths_.find(path);

                    if(elem == childGraph->criticalPaths_.end())
                    {
                        //todo:: raise exception
//                        cout<<"ERROR ERROR COULD NOT FIND CRIPATH IN FUSION" << endl;
//                        cout<<"searchingFor p " << path.print(); cout<< endl;
//                        for(auto m : childGraph.criticalPaths_)
//                            cout<<"Have " << m.first.print(); cout << endl;
                    }
//                    cout<<"Found " ; path.print(); cout << endl;

                    //todo: keep criNodes
                    auto& criPath = elem->second;
                    auto newEdge = internalGraph_.edge_insert(parent,child,MetaEdge(criPath.cost));
                    (*newEdge).extend(*criPath.nodes);

                    edgeCostMap.insert(pair<OptimizationGraph::edge_list_iterator, double>(newEdge, criPath.cost));
                }
            //update our new last level
            lastLevelIdx_ = levels.size() -1;
        }
        //TODO::TODO::TOVERYDO:: this here routine assumes that the subgraph joining by the higher layer will be done
        //in top-down order. need to add the cases of  :
        // - found child but did not find parent.
        // - did not find parent nor child, but the graph is not empty ()
        // - may have multiple "last" levels, in case we implement the edgeRemoving in the main manager.
    }
}

void MetaGraph::solve()
{
    auto& firstLevel = levels[firstLevelIdx_];
    auto& lastLevel  = levels[lastLevelIdx_];
    bool foundPath = false;
    for( auto startNode : firstLevel.level)
        for( auto endNode : lastLevel.level)
        {
            //for each pair of starting and ending node inside the metaGraph, we will have a set of cri-paths.
            CriticalEdges criticalEdges;

            criticalEdges = dijkstra<StrategySet&,
                                        MetaEdge,
                                        costNodeIteratorComp,
                                        costEdgeIteratorComp,
                                        double>
                                    (internalGraph_,
                                        startNode,
                                        endNode,
                                        edgeCostMap);

            // Handle inf edge removal case
            if (criticalEdges.size() < 1)
            {
                // If failed dijkstra because of a missing inf edge, force a criticalPath of inf cost
                StrategySetPair newPair(&(*startNode), &(*endNode));
                auto criticalNodes = std::make_shared<CriticalPathNodes>();
                CriticalPath newPath(criticalNodes, std::numeric_limits<double>::infinity());
                criticalPaths_[newPair] = newPath;
                continue;
            }

            auto criticalNodes = make_shared<CriticalPathNodes>(criticalEdges.size()-1);
            double cost = 0;

            //we will not store the actual critical edges, rather we will store the critical nodes, since the node
            //is the element that actually has the strategy. We will not actually store the node, but a pointer to it)
            int ctr;
            for(ctr = 0; ctr < (int)criticalEdges.size() - 1; ++ctr)
            {
                auto edge = criticalEdges[ctr];
                (*criticalNodes)[ctr] = edge->sink();
                cost = (*edge) + cost;
            }
            cost = (*criticalEdges[ctr]) + cost;

            if(cost < numeric_limits<double>::infinity())
                foundPath = true;

            //todo:: do we really need to always copy this? but indeed we need to inherit existing edges
            for(const auto edge : criticalEdges)
            {
                const auto& edgeCriPath = (*edge).criticalPaths();
                criticalNodes->insert(criticalNodes->end(),edgeCriPath.begin(),edgeCriPath.end());
            }

            if(cost < numeric_limits<double>::infinity())
                foundPath = true;

            StrategySetPair newPair(&(*startNode),&(*endNode));
            CriticalPath newPath(criticalNodes,cost);

//                cout<<"Generated for " << (*startNode)["name"].get<string>() <<
//                        " id " << (*startNode)["id"].get<int>() <<
//                        " ==> " <<(*endNode)["name"].get<string>() <<
//                        " id " << (*startNode)["id"].get<int>() <<
//                        " cost : " << cost << endl;

            criticalPaths_[newPair] = newPath;
        }

    name = firstLevel.op->getName() +"==>"+levels[1].op->getName() + "==>" + lastLevel.op->getName();
    if(!foundPath)
        cout << "INFINITE PATH: " << name << endl;
    if(!foundPath){
    //    cout<< "ONLY INFINITE PATHS at " << name << endl;
        // TODO throw exception
    }
    solved_ = true;
}

shared_ptr<MetaGraph::CriticalPath> MetaGraph::getLowestCriticalPathExtended()
{
    //this function is a hack-work-around. Normally we would only need the critical path with the lowest cost in the final
    //stage. But we need to include the Input/Output node strategies too.... Normally we would not want this, since by
    //principle they should be unique. But currently SplitOverHOverlapped needs to come from the input.... This needs revised

    auto& firstLevel = levels[firstLevelIdx_].level;
    auto& lastLevel = levels[lastLevelIdx_].level;
    auto extendedPath = make_shared<CriticalPath>();
    extendedPath->cost = numeric_limits<double>::infinity();
    extendedPath->nodes = make_shared<CriticalPathNodes>();

    double bestCost = numeric_limits<double>::infinity();
    StrategySetPair bestPair;
    OptimizationGraph::node_list_iterator bestSource,bestSink;
    bool foundPath = false;
    
    for(auto parent : firstLevel)
        for(auto child : lastLevel)
        {
            const StrategySetPair path(&(*parent),&(*child));
            auto elem = criticalPaths_.find(path);

            if(elem == criticalPaths_.end())
            {
                //todo:: raise exception
                // cout<<"ERROR ERROR COULD NOT FIND CRIPATH IN FUSION" << endl;
//                cout<<"searchingFor p " ; path.print(); cout<< endl;
//                for(auto m : criticalPaths_)
//                    cout<<"Have " ; elem->first.print(); cout << endl;
            }
//            cout<<"Found " ; path.print(); cout << endl;

            auto& criPath = elem->second;
            if(criPath.cost < bestCost)
            {
                bestCost = criPath.cost;
                bestPair = path;
                bestSource = parent;
                bestSink = child;
                foundPath = true;
            }
        }
    if(!foundPath){
    //    cout<< "Unable to find non-infinite path between pivot nodes"<< endl;
        // TODO throw exception
    }

    auto& bestCriPath = criticalPaths_.find(bestPair)->second;

    extendedPath->cost = bestCriPath.cost;
    extendedPath->nodes = bestCriPath.nodes;
    extendedPath->nodes->push_back(bestSource);
    extendedPath->nodes->push_back(bestSink);

    return extendedPath;
}

void MetaGraph::write(string dotFileLocation,bool skipInf)
{
    ofstream ostream;

    utils::validatePath(dotFileLocation);

    string newName = name;
    std::replace(newName.begin(),newName.end(),'/','_');
    string outputFile = dotFileLocation  + newName + ".dot";
    ostream.open(outputFile, std::ios::trunc | std::ios::out);

    if (!ostream.is_open())
    {
        //todo::throw exceptions
        throw ArgumentError("MetaGraph", "output", dotFileLocation, "Unable to open output file");
    }

    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for(auto node = internalGraph_.node_begin(); node != internalGraph_.node_end(); ++ node)
    {
        const auto& nodeName = (*node)["name"].get<string>();
        const auto& nodeId = (*node)["id"].get<int>();
        std::string nodeDef = "\t\"" + nodeName + "_" + to_string(nodeId)+ "\" [shape=box,";

        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"> \
                    <TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"10.0\"><B>"
                    + nodeName + "_" + to_string(nodeId)
                    + "</B></FONT></TD></TR>";

        for(const auto strategy : (*node))
        {
            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.first
                        + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.second.toString() + "</FONT></TD></TR>";
        }

        nodeDef += "</TABLE>>";
        ostream << nodeDef << "];\n";
    }

    for(auto edge = internalGraph_.edge_begin(); edge != internalGraph_.edge_end(); ++edge)
    {
        auto cost = (*edge).cost();
        if( skipInf and ( cost == numeric_limits<double>::infinity()))
            continue;

        auto sourceNode = edge->source();
        auto sinkNode = edge->sink();

        const auto& sourceNodeName = (*sourceNode)["name"].get<string>();
        const auto& sourceNodeId = (*sourceNode)["id"].get<int>();
        const auto& sinkNodeName = (*sinkNode)["name"].get<string>();
        const auto& sinkNodeId = (*sinkNode)["id"].get<int>();

        std::string edgeDef = "\t\""
                            + sourceNodeName + "_" + to_string(sourceNodeId)
                            + "\" -> \""
                            + sinkNodeName + "_" + to_string(sinkNodeId)
                            + "\"";

        edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" \
                    CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"> \
                    <FONT POINT-SIZE=\"14.0\"><B>" \
                    + std::to_string(cost)
                    + "</B></FONT></TD></TR>";

        edgeDef += "</TABLE>>];";

        ostream << edgeDef << "\n";
    }

    ostream << "}\n";
    ostream.close();
}

}
}
