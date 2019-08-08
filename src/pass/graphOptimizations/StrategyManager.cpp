#include "limits"
#include "tuple"

#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
//#include "include/mcm/pass/graphOptimizations/descartes_product.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"
#include "include/mcm/base/element.hpp"
#include "include/mcm/algorithms/dijkstra.hpp"

namespace mv {
namespace graphOptimizer {

using namespace std;

StrategyManager::StrategyManager(OpModel& model,mv::Element& passDesc) :
        model_(model),passDesc_(passDesc)
{

}

void StrategyManager::updateValuesFromJSON()
{
    auto graphOptimizerConfig = passDesc_.get<mv::Element>("graphOptimizerConfig");


    auto globalConfigs = graphOptimizerConfig.get<vector<mv::Element>>("globalConfigs");
    auto globalStrategies = graphOptimizerConfig.get<vector<mv::Element>>("globalStrategies");
    auto layerStrategySets  = graphOptimizerConfig.get<vector<mv::Element>>("layerStrategies");

    for( auto globalConfig : globalConfigs)
    {
        auto configName = globalConfig.getName();
        auto configValue = globalConfig.get("value");
        globalConfig_[configName] = configValue;
    }

    for( auto globalStrategy : globalStrategies)
    {
        auto strategyName = globalStrategy.getName();
        auto strategyValue = globalStrategy.get("value");
        globalStrategies_[strategyName] = strategyValue;
    }

    for( auto layerStrategySet : layerStrategySets)
    {
        auto layerName = layerStrategySet.getName();
        auto strategySets = layerStrategySet.get<vector<mv::Element>>("strategies");

        for(auto strategySet : strategySets)
        {
            auto strategySetName = strategySet.getName();
//            auto strategies = strategySet.get<vector<string>>("value");

            auto strategyValue = strategySet.get("value");
            layerStrategies_[layerName][strategySetName] = strategyValue;
//            if(strategiesType == typeid(vector<string>))
//            {
//                auto strategies = strategySet.get<vector<string>>("value");
//                for( auto strategy : strategies)
//                {
//                    layerStrategies_[layerName][strategySetName].insert(strategy);
//                }
//            }
//            else
//            {
//                layerStrategies_[layerName][strategySetName].insert(strategySet.get("value"));
//            }
        }
    }
}

void StrategyManager::updateDefaultValues()
{
    //TODO:: solve the "multiple registry" problem
    auto& globalConfigRegistry = mv::graphOptimizer::GlobalConfigRegistry::instance();
    auto& globalStrategyRegistry = mv::graphOptimizer::GlobalStrategyRegistry::instance();
    auto& layerStrategyRegistry = mv::graphOptimizer::LayerStrategyRegistry::instance();

    for (const auto& globalConfigName : globalConfigRegistry.list())
    {
       if(globalConfig_.find(globalConfigName) == globalConfig_.end() )
       {
           auto configVal = globalConfigRegistry.find(globalConfigName)->getAttr();
           globalConfig_[globalConfigName] = configVal;
       }
    }

    for (const auto& globalStrategyName : globalStrategyRegistry.list() )
    {
        if(globalStrategies_.find(globalStrategyName) == globalStrategies_.end())
        {
            auto strategyVal = globalStrategyRegistry.find(globalStrategyName)->getAttr();
            globalStrategies_[globalStrategyName] = strategyVal;
        }
    }

    for(auto& layer : layerStrategyRegistry.list())
    {
        auto strategySet = layerStrategyRegistry.find(layer)->getStrategySet();
        auto recordedStrategySet = layerStrategies_.find(layer);
        if(recordedStrategySet == layerStrategies_.end())
        {
            layerStrategies_[layer] = strategySet;

        }
        else
        {
            for(const auto& strategy : strategySet)
            {
                if( recordedStrategySet->second.find(strategy.first) == recordedStrategySet->second.end())
                {
                    layerStrategies_[layer][strategy.first] = strategy.second;
                }
            }
        }
    }

}

void StrategyManager::printStrategy()
{
    cout <<"########## Final Strategy Config" << endl;

    cout <<"Global Configs" << endl;
    for( const auto elem : globalConfig_)
    {
        cout <<"\t"<<elem.first << " : " << elem.second.toString() << endl;
    }

    cout <<"Global Strategies" << endl;
    for( const auto elem : globalStrategies_)
    {
        cout <<"\t"<<elem.first << " : " << elem.second.toString() << endl;
    }

    cout <<"LayerWise Strategies" << endl;
    for( const auto layer : layerStrategies_)
    {
        cout <<"\t"<<layer.first <<endl;

        for (const auto strategySet : layer.second )
        {
            cout <<"\t"<<strategySet.first << " : " << strategySet.second.toString() << endl;
//            cout <<"\t " << strategySet.first << "[ ";
//            for( const auto strategy : strategySet.second)
//            {
//                cout << strategy.toString() <<" ";
//            }
//            cout << "]" << endl;
        }
    }
}


//TODO:: error if the strategy is not there...
Attribute& StrategyManager::getStrategy(mv::Op op,string strategy)
{
    auto layerEntry = layerStrategies_.find(op.getName());

    if(layerEntry == layerStrategies_.end())
    {
        layerEntry = layerStrategies_.find(op.getOpType());
    }

    if(layerEntry == layerStrategies_.end())
        throw LogicError(*this, "could not find strategy entry for " + op.getName());

    auto& layerCfg = layerEntry->second;

    auto strategyEntry = layerCfg.find(strategy);
    if(strategyEntry == layerCfg.end())
    {
        strategyEntry = globalStrategies_.find(strategy);
    }

//    cout<< "for op: " << op.getName() << " str: " << strategy << " got: " << strategyEntry->second.toString() << endl;
    return strategyEntry->second;
}

void StrategyManager::writeDot(mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>& optimizationGraph, bool skipInf)
{
    ofstream ostream;
    string outputFile = dotFileLocation;

    ostream.open(outputFile,ios::trunc | ios::out);

    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for(auto node = optimizationGraph.node_begin(); node != optimizationGraph.node_end(); ++ node)
    {
        std::string nodeDef = "\t\"" + get<1>(*node)["name"].get<string>()  + "_" + to_string((long long unsigned)(void*)&(*node)) + "\" [shape=box,";
//        nodeDef += " label=\"" + (*node)["name"].toString() + "\\n";
        //TODO:: using an object's address to uniquely identify it is a baaaaaaaaad idea. Come up with something normal
        //basically, in our graph, we can have multiple nodes with the same name, but cannot have that in the dotfile
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"> \
                    <TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"10.0\"><B>"
                    + get<1>(*node)["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*node))
                    + "</B></FONT></TD></TR>";
        for(const auto strategy : get<1>(*node))
        {
//            nodeDef += strategy.first + ": " + strategy.second.toString() + "\\n";
            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.first
                        + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.second.toString() + "</FONT></TD></TR>";
        }

//        nodeDef += "\"";
        nodeDef += "</TABLE>>";

        ostream << nodeDef << "];\n";
    }

    for(auto edge = optimizationGraph.edge_begin(); edge != optimizationGraph.edge_end(); ++edge)
    {
//        if( skipInf and ( (*edge) == numeric_limits<double>::infinity()))
skipInf=false;
        if( skipInf and ( (*edge) == 999999.999))
            continue;
        //TODO:: using an object's address to uniquely identify it is a baaaaaaaaad idea. Come up with something normal
        std::string edgeDef = "\t\""
                            + get<1>(*(edge->source()))["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*(edge->source())))
                            + "\" -> \""
                            + get<1>(*(edge->sink()))["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*(edge->sink())))
                            + "\"";

        edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" \
                    CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"> \
                    <FONT POINT-SIZE=\"14.0\"><B>" \
                    + std::to_string(*edge)
                    + "</B></FONT></TD></TR>";

        edgeDef += "</TABLE>>];";

        ostream << edgeDef << "\n";
    }

    ostream << "}\n";
    ostream.close();
}

void StrategyManager::saveStrategy(std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator> cPathEdges)
{
  
    cout << "Saving streaming Strategy to Compilation Descriptor" << endl;
    cout << "    Critical Path info: length = " << cPathEdges.size() << endl ;
    for (int showPath=0; showPath<cPathEdges.size(); showPath++){
        cout << "    edge_"<< showPath << " cost is " << *(cPathEdges[showPath]) << endl ;
        cout << "    source is: " << get<1>(*cPathEdges[showPath]->source()).begin()->first << endl;
    }
    auto globalParams = model_.getGlobalConfigParams();
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    //determine if node already has strategy from JSON text, do not override text specification
    std::map<std::string, bool> hasSpec;

    for (auto s : strategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i = 0; i < splitList.size(); i++)
        {
            if ((splitList[i].hasAttr("C"))||(splitList[i].hasAttr("H"))||(splitList[i].hasAttr("W"))||(splitList[i].hasAttr("K")))
                hasSpec.insert(std::pair<std::string, bool>(nodeName, true));
            else
                hasSpec.insert(std::pair<std::string, bool>(nodeName, false));
        }
    }
      
    //save streaming strategy into compilation descriptor
    auto copyElement = strategyList[0];
    auto copyName = copyElement.get<std::string>("name_filter");
    auto copySplits =  copyElement.get<std::vector<mv::Element>>("splits");
    for (int i=copySplits.size(); i<4; i++)
        copySplits.push_back(copySplits[0]);    // 4 element vector for streaming strategies c,h,w,k
    for (int savePath=1; savePath<cPathEdges.size(); savePath++)
    {
        auto parentNode = cPathEdges[savePath]->leftmost_parent()->sink();
        auto parentStrategySet = get<1>(*parentNode) ;
        mv:Shape newStrategy = parentStrategySet["streaming"];
        std::string newName = parentStrategySet["name"] ;
        if ( hasSpec.find(newName) == hasSpec.end())
        { 
            copyElement.set("name_filter",newName);
            copySplits[0].set<int>("C", newStrategy[0]);
            copySplits[1].set<int>("H", newStrategy[1]);
            copySplits[2].set<int>("W", newStrategy[2]);
            copySplits[3].set<int>("K", newStrategy[3]);
            copyElement.set("splits",copySplits);
            strategyList.push_back(copyElement);
        }
    }
    globalParams->set("streaming_strategy", strategyList);

    //test updated compilation descriptor
    auto globalParams2 = model_.getGlobalConfigParams();
    auto strategyList2 = globalParams2->get<std::vector<mv::Element>>("streaming_strategy");

    for (auto s : strategyList2)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        std::cout <<" Streaming strategy (from compilation descriptor) for node " << s.get<std::string>("name_filter") <<  std::endl ;
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i = 0; i < splitList.size(); i++)
        {
            if (splitList[i].hasAttr("C"))
            {
                std::cout << "     C : " << splitList[i].get<int>("C") << std::endl;
            }
            else if (splitList[i].hasAttr("H"))
            {
                std::cout << "     H : " << splitList[i].get<int>("H") << std::endl;
            }
            else if (splitList[i].hasAttr("W"))
            {
                std::cout << "     W : " << splitList[i].get<int>("W") << std::endl;
            }
            else if (splitList[i].hasAttr("K"))
            {
                std::cout << "     K : " << splitList[i].get<int>("K") << std::endl;
            }
        }
    }
}

void StrategyManager::saveMetaStrategy(std::vector<MetaGraph::edge_list_iterator> cPathEdges)
{
  
    cout << "Saving streaming Strategy to Compilation Descriptor" << endl;
    /* cout << "    Critical Path info: length = " << cPathEdges.size() << endl ;
    for (int showPath=0; showPath<cPathEdges.size(); showPath++){
        cout << "    edge_"<< showPath << " cost is " << *(cPathEdges[showPath]) << endl ;
        cout << "    source is: " << get<1>(*cPathEdges[showPath]->source()).begin()->first << endl;
    }*/
    vector<StrategySet> allStrategies;
    for(auto edge : cPathEdges){
        for(auto strategy : (*edge).second){
            allStrategies.push_back(strategy);
        }
    }

    auto globalParams = model_.getGlobalConfigParams();
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    //determine if node already has strategy from JSON text, do not override text specification
    std::map<std::string, bool> hasSpec;

    for (auto s : strategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i = 0; i < splitList.size(); i++)
        {
            if ((splitList[i].hasAttr("C"))||(splitList[i].hasAttr("H"))||(splitList[i].hasAttr("W"))||(splitList[i].hasAttr("K")))
                hasSpec.insert(std::pair<std::string, bool>(nodeName, true));
            else
                hasSpec.insert(std::pair<std::string, bool>(nodeName, false));
        }
    }
      
    //save streaming strategy into compilation descriptor
    auto copyElement = strategyList[0];
    auto copyName = copyElement.get<std::string>("name_filter");
    auto copySplits =  copyElement.get<std::vector<mv::Element>>("splits");
    for (int i=copySplits.size(); i<4; i++)
        copySplits.push_back(copySplits[0]);    // 4 element vector for streaming strategies c,h,w,k
    for (auto strategy : allStrategies)
    {
        mv:Shape newStrategy = strategy["streaming"];
        std::string newName = strategy["name"] ;
        if ( hasSpec.find(newName) == hasSpec.end())
        { 
            copyElement.set("name_filter",newName);
            copySplits[0].set<int>("C", newStrategy[0]);
            copySplits[1].set<int>("H", newStrategy[1]);
            copySplits[2].set<int>("W", newStrategy[2]);
            copySplits[3].set<int>("K", newStrategy[3]);
            copyElement.set("splits",copySplits);
            strategyList.push_back(copyElement);
        }
    }
/*    for (int savePath=1; savePath<cPathEdges.size(); savePath++)
    {
        auto parentNode = cPathEdges[savePath]->leftmost_parent()->sink();
        auto parentStrategySet = get<1>(*parentNode) ;
        mv:Shape newStrategy = parentStrategySet["streaming"];
        std::string newName = parentStrategySet["name"] ;
        if ( hasSpec.find(newName) == hasSpec.end())
        { 
            copyElement.set("name_filter",newName);
            copySplits[0].set<int>("C", newStrategy[0]);
            copySplits[1].set<int>("H", newStrategy[1]);
            copySplits[2].set<int>("W", newStrategy[2]);
            copySplits[3].set<int>("K", newStrategy[3]);
            copyElement.set("splits",copySplits);
            strategyList.push_back(copyElement);
        }
    }
    */
    globalParams->set("streaming_strategy", strategyList);

    //test updated compilation descriptor
    auto globalParams2 = model_.getGlobalConfigParams();
    auto strategyList2 = globalParams2->get<std::vector<mv::Element>>("streaming_strategy");

    for (auto s : strategyList2)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        std::cout <<" Streaming strategy (from compilation descriptor) for node " << s.get<std::string>("name_filter") <<  std::endl ;
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i = 0; i < splitList.size(); i++)
        {
            if (splitList[i].hasAttr("C"))
            {
                std::cout << "     C : " << splitList[i].get<int>("C") << std::endl;
            }
            else if (splitList[i].hasAttr("H"))
            {
                std::cout << "     H : " << splitList[i].get<int>("H") << std::endl;
            }
            else if (splitList[i].hasAttr("W"))
            {
                std::cout << "     W : " << splitList[i].get<int>("W") << std::endl;
            }
            else if (splitList[i].hasAttr("K"))
            {
                std::cout << "     K : " << splitList[i].get<int>("K") << std::endl;
            }
        }
    }
}

void StrategyManager::linearDijkstra(mv::Data::OpListIterator opBegin)
{
    struct costEdgeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };

    struct costNodeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator rhs) const
        {
            int x = get<2>(*lhs) ;
            int y = get<2>(*rhs) ;
            return (x < y) ;
        }
    };

    mv::graph<std::tuple<mv::Op&,StrategySet,int>,double> optimizationGraph;
    vector<vector<StrategySet>> nodeStrategies;
    vector<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator> old_nodes,new_nodes;

    old_nodes.clear();

    map<typename graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double, costEdgeIteratorComp> edgeCostMap;
    int opCtr = 0;
    int optionCtr = 0;
    
    for(auto op = opBegin; op != model_.opEnd(); ++op)
    {
        if(op->getOpType() == "ConstantInt")
            continue;
        vector<StrategySet> nodeStrategy;

        opCtr++;

        generateStrategySetForLayer(*op,nodeStrategy);
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*op,strategy,optionCtr++)));
        }

        for(const auto oldNode : old_nodes)
            for(const auto newNode : new_nodes)
            {
//                cout<< "In linearDykstra: inserting edge to optGraph from " << get<1>(*oldNode)["name"].get<string>()  << "_"<< to_string((long long unsigned)(void*)&(*oldNode)) << " to " << get<1>(*newNode)["name"].get<string>()  << "_"<< to_string((long long unsigned)(void*)&(*newNode)) << endl;
                double edgeCost = transitionCost( get<0>(*oldNode),
                                                  get<0>(*newNode),
                                                  get<1>(*oldNode),
                                                  get<1>(*newNode));
                int edgeCostInt = edgeCost ;
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                edgeCostMap.insert(std::pair<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double>(newEdge, edgeCost));
 //               cout<< "        cost = " << edgeCost << endl;
            }
        old_nodes.swap(new_nodes);

        nodeStrategies.push_back(nodeStrategy);
    }

    writeDot(optimizationGraph,true);
    costEdgeIteratorComp costEdgeCompare;

    auto sinkNodeIt = optimizationGraph.node_begin() ;
    for (int ii=0; ii<optimizationGraph.node_size()-1; ii++) ++sinkNodeIt;

    auto criticalPathEdges = dijkstra<std::tuple<mv::Op&,StrategySet,int>,double,costNodeIteratorComp,costEdgeIteratorComp, double>(optimizationGraph,optimizationGraph.node_begin(),sinkNodeIt,edgeCostMap);
    
    saveStrategy(criticalPathEdges);
}


void StrategyManager::recursiveDijkstra(mv::Data::OpListIterator opBegin)
{
    struct costEdgeIteratorComp
    {
        bool operator()(const MetaGraph::edge_list_iterator lhs, const MetaGraph::edge_list_iterator rhs) const
        {
            double x = get<0>(*lhs);
            double y = get<0>(*rhs);
            return (x < y);
        }
    };

    struct costNodeIteratorComp
    {
        bool operator()(const MetaGraph::node_list_iterator lhs, const MetaGraph::node_list_iterator rhs) const
        {
            int x = get<2>(*lhs) ;
            int y = get<2>(*rhs) ;
            return (x < y) ;
        }
    };
    cout << "calling extract subgraphs on the data graph" << endl ;
    std::unordered_set<std::string> recursedNodes;
    //Each vector of edges corresponds to a linear portion of the graph
    //pair<vector<OptimizationPair>, StrategySet> output = 
    MetaGraph metaGraph;
    recursiveCriticalPath(opBegin, recursedNodes, metaGraph);
    cout << " calling dijkstra on the meta graph" << endl;
    //iterate through all metaGraph nodes and make note of which are sources and which are sinks
    vector<MetaGraph::node_list_iterator> sources, sinks;
    for (auto it = metaGraph.node_begin(); it != metaGraph.node_end(); ++it)
    {
        if(it->parents_size() == 0){
            sources.push_back(it);
        }
        if(it->children_size() == 0){
            sinks.push_back(it);
        }
    }
    map<typename MetaGraph::edge_list_iterator, double, costEdgeIteratorComp> edgeCostMap;
    //build edge cost map needed to call dijkstra
    for (auto it = metaGraph.edge_begin(); it != metaGraph.edge_end(); ++it){
        edgeCostMap.insert(std::pair<MetaGraph::edge_list_iterator, double>(it, get<0>(*it)));
    }
    //call dijkstra on metagraph for each source and sink combo, choosing the best
    double max = 99999999.99;
    vector<MetaGraph::edge_list_iterator> finalCriticalPath;
    for(auto source : sources){
        for(auto sink : sinks) {
            //call dijkstra here, if cost is less than max found so far, save otherwise move on
            vector<MetaGraph::edge_list_iterator> criticalPathEdges = dijkstra<std::tuple<mv::Op&,StrategySet,int>,MetaGraphEdge,costNodeIteratorComp,costEdgeIteratorComp, double>(metaGraph,source,sink,edgeCostMap);
            double cost = 0;
            for (auto edge : criticalPathEdges){
                cost += get<0>(*edge);
            }
            if(cost < max){
                finalCriticalPath = criticalPathEdges;
            }
        }
    }
    //save best found strategy
    saveMetaStrategy(finalCriticalPath);
}

void StrategyManager::recursiveCriticalPath(typename graph<mv::Op, mv::DataFlow>::node_list_iterator modelSource, 
                                            std::unordered_set<std::string>& recursedNodes, MetaGraph& metaGraph){
    struct costEdgeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };

    struct costNodeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator rhs) const
        {
            int x = get<2>(*lhs) ;
            int y = get<2>(*rhs) ;
            return (x < y) ;
        }
    };

    vector<vector<CriticalInfo>> parallelLinearSections;
    vector<CriticalInfo> parallelLinearSection;
    vector<OptimizationGraph> allOptimizationGraphs;
    
    mv::graph<mv::Op, mv::DataFlow> g;
    vector<typename graph<mv::Op, mv::DataFlow>::node_list_iterator> next_modelSource;

    //BASE CASE - end of graph, or previously recursed on this source
    std::string opName = (*modelSource).getName();
    if(modelSource->leftmost_child() == g.node_end() || recursedNodes.find(opName) != recursedNodes.end() || (*modelSource).getOpType() == "ConstantInt"){
        return;
    }
    int childCtr = 0;
    recursedNodes.insert(opName); //haven't seen this source before, save it
    //RECURSIVE CASE - iterate over the children of source build linear subgraphs, kick off recursion if another pivot node hit
    for(auto model_edge  = modelSource->leftmost_output(); model_edge !=  g.edge_end(); ++model_edge)
    {
        if((*(model_edge->sink())).getOpType() == "ConstantInt")
            continue;
        childCtr++;
         //Create the subgraph to pass to dijkstra, build iteratively until hit next pivot node
        OptimizationGraph optimizationGraph;
        vector<vector<StrategySet>> nodeStrategies;
        vector<OptimizationGraph::node_list_iterator> old_nodes, new_nodes, first_nodes, last_nodes;

        old_nodes.clear();

        map<typename OptimizationGraph::edge_list_iterator, double, costEdgeIteratorComp> edgeCostMap;
        int opCtr = 0;
        int optionCtr = 0;

        //Add modelSource (start pivot) to the graph
        vector<StrategySet> nodeStrategy;
        opCtr++;
        generateStrategySetForLayer(*modelSource,nodeStrategy);
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*modelSource,strategy,optionCtr++)));
        }
        first_nodes = new_nodes;
        for(const auto oldNode : old_nodes)
            for(const auto newNode : new_nodes)
            {
                double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                int edgeCostInt = edgeCost ;
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                edgeCostMap.insert(std::pair<OptimizationGraph::edge_list_iterator, double>(newEdge, edgeCost));
            }
        old_nodes.swap(new_nodes);
        nodeStrategies.push_back(nodeStrategy);

        auto model_child = model_edge->sink();
        auto model_parent = model_edge->source();

        //ITERATE through linear section, building the optimization subgraph, while (model_child is not a pivot node)
        while ( model_child->children_size() == 1 && ( model_child->parents_size() == 1 || ( model_child->parents_size() == 2 && 
                ( (*model_child->leftmost_input()->source()).getOpType() == "ConstantInt"  || 
                (*model_child->rightmost_input()->source()).getOpType() == "ConstantInt" ) ))) 
        {
            vector<StrategySet> nodeStrategy;
            opCtr++;

            generateStrategySetForLayer(*model_child,nodeStrategy);
            new_nodes.clear();
            for(auto strategy : nodeStrategy)
            {
                new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*model_child,strategy,optionCtr++)));
            }

            for(const auto oldNode : old_nodes)
                for(const auto newNode : new_nodes)
                {
                    double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                    int edgeCostInt = edgeCost ;
                    auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                    edgeCostMap.insert(std::pair<OptimizationGraph::edge_list_iterator, double>(newEdge, edgeCost));
                }
            old_nodes.swap(new_nodes);

            nodeStrategies.push_back(nodeStrategy);

            //move to next child node in this linear section
            model_child = model_child->leftmost_output()->sink();
        }
        //Iteration ends when next pivot node is found, include this pivot node in the graph
        nodeStrategy.clear();
        opCtr++;
        generateStrategySetForLayer(*model_child,nodeStrategy);
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*model_child,strategy,optionCtr++)));
        }
        for(const auto oldNode : old_nodes)
            for(const auto newNode : new_nodes)
            {
                double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                int edgeCostInt = edgeCost ;
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                edgeCostMap.insert(std::pair<OptimizationGraph::edge_list_iterator, double>(newEdge, edgeCost));
            }
        old_nodes.swap(new_nodes);
        nodeStrategies.push_back(nodeStrategy);
        last_nodes = old_nodes;

/*
    Next call dijkstra on the each linear subgraph found from the modelSource on this recursive pass
 */
        allOptimizationGraphs.push_back(optimizationGraph);
        writeDot(optimizationGraph,true);
        costEdgeIteratorComp costEdgeCompare;
        auto sinkNodeIt = optimizationGraph.node_begin() ;
        for (int ii=0; ii<optimizationGraph.node_size()-1; ii++) ++sinkNodeIt;

        //Critical Path is node iter source, node iter sink, vector of edge iters critical path, double edge cost sum
        //cout << "Found starting and endings nodes: " << first_nodes.size() << ", " << last_nodes.size() << endl;
        for(auto startingNode : first_nodes){
            for( auto endingNode : last_nodes)
            {
                CriticalEdges criticalPathEdges = dijkstra<std::tuple<mv::Op&,StrategySet,int>,double,costNodeIteratorComp,costEdgeIteratorComp, double>(optimizationGraph,startingNode,endingNode,edgeCostMap);
                double cost = 0;
                vector<StrategySet> strategies;
                for(auto edge : criticalPathEdges){
                    cost += *edge; //TODO add first source strategy
                    strategies.push_back(get<1>(*edge->sink()));
                }
                CriticalInfo particulars = CriticalInfo(startingNode, endingNode, MetaGraphEdge(cost, strategies));
                parallelLinearSection.push_back(particulars);
            }
        }
        parallelLinearSections.emplace_back(parallelLinearSection);
        cout << "done with this child" << endl;
    //TODO add check here for whether we've hit the same pivot (needed for nested parallelism)
    //recurse if we haven't all hit the same pivot (into the nested parallel branch)
        next_modelSource.push_back(model_child);
    }

    //If child counter is 1, we were in a linear section, just add the subgraph to the big graph and move on
    if(childCtr == 1){
        vector<CriticalInfo> linearSection = parallelLinearSections[0]; //Vector of Critical Info, add this to the metagraph
        for(CriticalInfo critInfo : linearSection){
            MetaGraph::node_list_iterator source, sink;
            if(metaGraph.node_size() == 0 ){
                source = metaGraph.node_insert(*get<0>(critInfo));
                sink = metaGraph.node_insert(*get<1>(critInfo));
            }else {
                source = metaGraph.node_find(*get<0>(critInfo));
                if(source == metaGraph.node_end()){ //source node does not exist in metaGraph add it
                    source = metaGraph.node_insert(*get<0>(critInfo));
                }
                sink = metaGraph.node_find(*get<1>(critInfo));
                if(sink == metaGraph.node_end()){ //sink node does not exist in metaGraph add it
                    sink = metaGraph.node_insert(*get<1>(critInfo));
                }
            }
            double cost = get<2>(critInfo).first;
            vector<StrategySet> strategies = get<2>(critInfo).second;
            metaGraph.edge_insert(source, sink, MetaGraphEdge(cost, strategies));
        }
    }
    //If child counter is greater than 1, we were in a parallel section and add the sum of all relevant edges to a new meta subgraph and add that subgraph to the big graph

/* 
    if(sinkReturnVector.empty()){
       // cout << "*** In Linear Graph or end of graph Linear Section" << endl;
        CriticalPair finalPair;
        double max = 99999999.999;
        OptimizationGraph optGraph = parallelLinearSections[0].first;

        for(auto criticalPath : parallelLinearSections[0].second)
        {
            double cost = 0;
            for (auto edge : criticalPath){
                cost += *edge;
            }
            //cout << "  Found cost of " << cost << " current  max= " << max << endl;
            if( cost < max)
            {
                finalPair = CriticalPair(optGraph, criticalPath);
                max = cost;
                //cout << "    Updated best cost to " << max << endl;
            }
        }

        saveStrategyGraph(finalPair);
        StrategySet strategy = get<1>(*finalPair.second.front()->source());
        //return pair<vector<OptimizationPair>, StrategySet>(parallelLinearSections, strategy);
    }
    //Linear section no branches, use found source strategy to lock the sink for criticalPaths and pick the best with this constraint (linear part of parallel graph)
    else if(childCtr == 1){
        //cout << "*** In Linear Section" << endl;
     	vector<OptimizationPair> returnVector = sinkReturnVector;
        StrategySet sinkStrategy = sinkReturnStrategy;

     	OptimizationPair thisSection = parallelLinearSections[0]; //only one linear part in this section
        OptimizationGraph optGraph = parallelLinearSections[0].first;

        CriticalEdges finalCriticalPath; 
        CriticalPair finalPair;
        StrategySet finalStrategy;

        double max = 99999999.999;

        //check first that this is a CP we should consider (does the sink match the strategy already decided for that node)
        for(auto criticalPath : thisSection.second)
        {
        	double cost = 0;
            for (int showPath=0; showPath < criticalPath.size(); showPath++){
                cost += *criticalPath[showPath];
            }
            // if strategy is a match we want this to execute
            if(sinkStrategy == get<1>(*criticalPath.front()->sink()) && cost < max)
            {
                finalPair = CriticalPair(optGraph, criticalPath);
                finalStrategy = get<1>(*criticalPath.front()->source());
                max = cost;
            }
        }
        saveStrategyGraph(finalPair);

        returnVector.push_back(parallelLinearSections[0]);
        //return pair<vector<OptimizationPair>, StrategySet>(returnVector, finalStrategy);
    }
    //Parallel branches between this source and sink, do addition to pick the best
     else{
        //cout << "*** In Parallel Section" << endl;
    	vector<OptimizationPair> returnVector = sinkReturnVector;
        StrategySet sinkStrategy = sinkReturnStrategy;

        vector<vector<CriticalPair>> sinkMatchedCriticalPaths;
        vector<vector<CriticalPair>> sourceMatchedCriticalPaths;

        vector<CriticalPair> bestFinalPairs;
        StrategySet bestStrategy;

        double max = 99999999.999;


        int numSources = 0;
        for(auto branch : parallelLinearSections){
            vector<CriticalPair> found;
            for(CriticalEdges path : branch.second){
                auto lastEdge = path.back();
                auto sinkNode = lastEdge->sink();
                StrategySet pathStrategy = get<1>(*sinkNode);
                if(pathStrategy == sinkStrategy){
                    found.push_back(CriticalPair(branch.first, path));
                }
            }
            sinkMatchedCriticalPaths.push_back(found);
            found.clear();
        }


        numSources = sinkMatchedCriticalPaths[0].size(); //Assume all paths have the same number of source options (it's the same original graph node)

        for(int source = 0; source < numSources; source++){
            double cost = 0;
            vector<CriticalPair> potentialFinalPairs;
            StrategySet sourceStrategy = get<1>(*sinkMatchedCriticalPaths[0].at(source).second.front()->source());
            for(auto branch : sinkMatchedCriticalPaths){
                for(auto path : branch){
                    if(get<1>(*path.second.front()->source()) == sourceStrategy){
                        potentialFinalPairs.push_back(path);
                        for(auto edge : path.second){
                            cost += *edge;
                        }
                        continue;
                    }
                }
            }
            if(cost < max)
            {
                max = cost;
                bestFinalPairs = potentialFinalPairs;
                bestStrategy = sourceStrategy;
            }
            potentialFinalPairs.clear();
        }

        for(auto criticalPair : bestFinalPairs){
        	saveStrategyGraph(criticalPair);
        }

        for(auto linearSection : parallelLinearSections){
            returnVector.push_back(linearSection);
        }
        //return pair<vector<OptimizationPair>, StrategySet>(returnVector, bestStrategy);
    }
*/
    for(auto source : next_modelSource)
    {
        std::string next_opName = (*source).getName();
        if((*source).getOpType() != "ConstantInt" && recursedNodes.find(next_opName) == recursedNodes.end()){
            cout << "Recursing on " << next_opName << endl;
            recursiveCriticalPath(source, recursedNodes, metaGraph);
        }
    }

}
void StrategyManager::generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec)
{
    //TODO:: error
    cout<<"ERROR generateStrategySetForLayer" << endl;
}

double StrategyManager::transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
{
    cout<<"ERROR transitionCost" << endl;
    return -1;
}
}
}