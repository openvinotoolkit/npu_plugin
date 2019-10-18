#include "limits"
#include "tuple"
#include "chrono"

#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"
#include "include/mcm/base/element.hpp"
#include "include/mcm/algorithms/dijkstra.hpp"
#include "include/mcm/utils/env_loader.hpp"

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

            auto strategyValue = strategySet.get("value");
            layerStrategies_[layerName][strategySetName] = strategyValue;
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
        }
    }
}


Attribute& StrategyManager::getStrategy(mv::Op op,string strategy)
{
    auto layerEntry = layerStrategies_.find(op.getName());

    if(layerEntry == layerStrategies_.end())
    {
        layerEntry = layerStrategies_.find(op.getOpType());
    }

    if(layerEntry == layerStrategies_.end())
        throw LogicError(*this, "StrategyManager could not find strategy entry for " + op.getName());

    auto& layerCfg = layerEntry->second;

    auto strategyEntry = layerCfg.find(strategy);
    if(strategyEntry == layerCfg.end())
    {
        strategyEntry = globalStrategies_.find(strategy);
    }

    return strategyEntry->second;
}

void StrategyManager::writeMetaDot(MetaGraph& graph, bool skipInf)
{
    ofstream ostream;

    utils::validatePath(dotFileLocation);

    string outputFile = dotFileLocation +"_finalMetaGraph.dot";
    ostream.open(outputFile, std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw ArgumentError(model_, "output", dotFileLocation, "Unable to open output file");


    ostream.open(outputFile,ios::trunc | ios::out);
    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for(auto node = graph.node_begin(); node != graph.node_end(); ++ node)
    {
        std::string nodeDef = "\t\"" + get<1>(*node)["name"].get<string>()  + "_" + to_string(get<2>(*node)) + "\" [shape=box,";
        //TODO:: using an object's address to uniquely identify it is a baaaaaaaaad idea. Come up with something normal
        //basically, in our graph, we can have multiple nodes with the same name, but cannot have that in the dotfile
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"> \
                    <TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"10.0\"><B>"
                    + get<1>(*node)["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*node))
                    + "</B></FONT></TD></TR>";
        for(const auto strategy : get<1>(*node))
        {
            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.first
                        + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.second.toString() + "</FONT></TD></TR>";
        }
        nodeDef += "</TABLE>>";

        ostream << nodeDef << "];\n";
    }

    for(auto edge = graph.edge_begin(); edge != graph.edge_end(); ++edge)
    {
        if( skipInf and ( get<0>(*edge) == inf_))
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
                    + std::to_string(get<0>(*edge))
                    + "</B></FONT></TD></TR>";

        edgeDef += "</TABLE>>];";

        ostream << edgeDef << "\n";
    }

    ostream << "}\n";
    ostream.close();
}

unsigned int globalCtr = 0;


void StrategyManager::writeDot(OptimizationGraph& optimizationGraph, bool skipInf)
{
    ofstream ostream;

    utils::validatePath(dotFileLocation);

    auto start_node = optimizationGraph.node_begin() ;
    auto start_node_name =  get<1>(*start_node)["name"].get<string>();
    std::replace(start_node_name.begin(),start_node_name.end(),'/','_');

    string outputFile = dotFileLocation + start_node_name  + "_" + to_string(globalCtr++) + ".dot";
    ostream.open(outputFile,ios::trunc | ios::out);
    if (!ostream.is_open())
        throw ArgumentError(model_, "output", dotFileLocation, "Unable to open output file");

    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for(auto node = optimizationGraph.node_begin(); node != optimizationGraph.node_end(); ++ node)
    {
        std::string nodeDef = "\t\"" + get<1>(*node)["name"].get<string>()  + "_" + to_string((long long unsigned)(void*)&(*node)) + "\" [shape=box,";
        //TODO:: using an object's address to uniquely identify it is a baaaaaaaaad idea. Come up with something normal
        //basically, in our graph, we can have multiple nodes with the same name, but cannot have that in the dotfile
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"> \
                    <TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"10.0\"><B>"
                    + get<1>(*node)["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*node))
                    + "</B></FONT></TD></TR>";
        nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">opType: </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                        + get<0>(*node).getOpType() + "</FONT></TD></TR>";
        for(const auto strategy : get<1>(*node))
        {
            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.first
                        + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.second.toString() + "</FONT></TD></TR>";
        }
        nodeDef += "</TABLE>>";

        ostream << nodeDef << "];\n";
    }

    for(auto edge = optimizationGraph.edge_begin(); edge != optimizationGraph.edge_end(); ++edge)
    {
        if( skipInf and ( get<0>(*edge)== inf_))
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
                    + std::to_string(get<1>(*edge)) + " : " + std::to_string(get<0>(*edge))
                    + "</B></FONT></TD></TR>";

        edgeDef += "</TABLE>>];";

        ostream << edgeDef << "\n";
    }

    ostream << "}\n";
    ostream.close();
}


std::vector<mv::Element> StrategyManager::convertStreamingStrategyToElement(std::vector<StrategySet> &strategiesToConvert, std::shared_ptr<mv::Element> compDesc)
{
    log(Logger::MessageType::Info, "GraphOptimizer: Converting Strategies found to Element");

    auto streamingStrategyList = compDesc->get<std::vector<mv::Element>>("streaming_strategy");

    //determine if node already has streaming strategy from JSON text, do not override text specification
    std::vector<std::string> hasSpec;
    for (auto s : streamingStrategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i = 0; i < splitList.size(); i++)
        {
            if ((splitList[i].hasAttr("C"))||(splitList[i].hasAttr("H"))||(splitList[i].hasAttr("W"))||(splitList[i].hasAttr("K")))
                hasSpec.push_back(nodeName);
        }
    }

    //cast streaming strategy into Element
    auto copyElement = streamingStrategyList[0];
    auto copyName = copyElement.get<std::string>("name_filter");
    auto copySplits =  copyElement.get<std::vector<mv::Element>>("splits");
    for (int i=copySplits.size(); i<4; i++)
        copySplits.push_back(copySplits[0]);    // 4 element vector for streaming strategies c,h,w,k
    for (auto strategy : strategiesToConvert)
    {

        mv::Shape newStrategy = strategy["streaming"];
        std::string newName = strategy["name"] ;
        if ( std::find(hasSpec.begin(), hasSpec.end(), newName) == hasSpec.end())
        {
            copyElement.set("name_filter",newName);
            copySplits[0].set<int>("C", newStrategy[0]);
            copySplits[1].set<int>("H", newStrategy[1]);
            copySplits[2].set<int>("W", newStrategy[2]);
            copySplits[3].set<int>("K", newStrategy[3]);
            copyElement.set("splits",copySplits);
            streamingStrategyList.push_back(copyElement);
        }
    }

    return streamingStrategyList;
}

std::vector<mv::Element> StrategyManager::convertClusteringStrategyToElement(std::vector<StrategySet> &strategiesToConvert, std::shared_ptr<mv::Element> compDesc)
{
    log(Logger::MessageType::Info, "GraphOptimizer: Converting Multiclustering Strategies to Element");

    auto clusteringStrategyList = compDesc->get<std::vector<mv::Element>>("split_strategy");

    //determine if node already has clustering strategy from JSON text, do not override text specification
    std::vector<std::string> hasClusterSpec;
    for (auto s : clusteringStrategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        std::string strategyName = s.get<std::string>("strategy");
        if ((strategyName=="SplitOverH") or (strategyName=="SplitOverK") or (strategyName=="SplitOverHOverlapped") or (strategyName=="HKSwitch"))
        {
            hasClusterSpec.push_back(nodeName);
        }
    }

    //save clustering strategy into compilation descriptor
    mv::Element copyCElement("");//= clusteringStrategyList[0];
    for (auto strategy : strategiesToConvert)
    {
        std::string newStrategy = strategy["clustering"];
        std::string newName = strategy["name"] ;
        if ( std::find(hasClusterSpec.begin(), hasClusterSpec.end(), newName) == hasClusterSpec.end())
        {
            copyCElement.set("name_filter",newName);
            copyCElement.set("strategy",newStrategy);
            clusteringStrategyList.push_back(copyCElement);
        }
    }

    return clusteringStrategyList;
}

std::vector<mv::Element> StrategyManager::convertLocationStrategyToElement(std::vector<StrategySet> &strategiesToConvert)
{
    log(Logger::MessageType::Info, "GraphOptimizer: Converting Location Strategies to Element");

    mv::Element copyLElement("");
    std::vector<mv::Element> locationStrategyList;

    for(auto strategy : strategiesToConvert)
    {
        auto spilling = strategy["spilling"].get<bool>();
        auto opName   = strategy["name"].get<string>();

        std::string DDRLocation = "DDR";
        std::string CMXLocation = "CMX";

        auto op = model_.getOp(opName);
        if(op->getOpType() == "Output")
            continue;

        if(spilling)
            copyLElement.set("mem_location",DDRLocation);
        else
            copyLElement.set("mem_location",CMXLocation);
        copyLElement.set("name_filter", opName);

        locationStrategyList.push_back(copyLElement);
    }

    return locationStrategyList;
}

std::vector<mv::Element> StrategyManager::convertSparsityStrategyToElement(std::vector<StrategySet> &strategiesToConvert){
    log(Logger::MessageType::Info, "GraphOptimizer: Converting Sparsity Strategies to Element");

    mv::Element copyLElement("");
    std::vector<mv::Element> sparsityStrategyList;

    for(auto strategy : strategiesToConvert)
    {
        auto inputActivationSparsity = strategy["inputSparsity"].get<bool>();
        auto outputActivationSparsity = strategy["outputSparsity"].get<bool>();
        auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
        auto opName   = strategy["name"].get<string>();

        auto op = model_.getOp(opName);

        copyLElement.set("inputActivationSparsity",inputActivationSparsity);
        copyLElement.set("outputActivationSparsity",outputActivationSparsity);
        copyLElement.set("weightsSparsity",weightsSparsity);
        copyLElement.set("name_filter", opName);

        sparsityStrategyList.push_back(copyLElement);
    }

    return sparsityStrategyList;
}

void StrategyManager::saveMetaStrategy(std::vector<MetaGraph::edge_list_iterator> cPathEdges)
{

    const bool enablePrintStrategyToTerminal = true;
    const bool enableSaveStrategyToDescriptor = true;
    const bool enableSaveStrategyToJsonFile = true;

    vector<StrategySet> allStrategies;
    for(auto edge : cPathEdges){
        allStrategies.push_back(get<1>(*edge->source())); //add strategy for each node in the metagraph that was chosen
        for(auto strategy : get<1>(*edge)){
            allStrategies.push_back(strategy); //all the nodes that were eliminated from metagraph get added
        }
    }

    auto lastNode = *cPathEdges.back()->sink();
    allStrategies.push_back(get<1>(lastNode)); //last node needs strategy too (one more node than edges)

    auto globalParams = model_.getGlobalConfigParams();

    std::vector<mv::Element> streamingStrategyElements = convertStreamingStrategyToElement(allStrategies, globalParams);
    std::vector<mv::Element> multiClusterStrategyElements = convertClusteringStrategyToElement(allStrategies, globalParams);
    std::vector<mv::Element> locationStrategyElements = convertLocationStrategyToElement(allStrategies);
    std::vector<mv::Element> sparsityStrategyElements = convertSparsityStrategyToElement(allStrategies);

    if (enableSaveStrategyToDescriptor)
    {
        log(Logger::MessageType::Info, "GraphOptimizer: Saving Strategy to Compilation Descriptor");
        auto compDesc = model_.getGlobalConfigParams();
        compDesc->set("streaming_strategy", streamingStrategyElements);
        compDesc->set("split_strategy", multiClusterStrategyElements);
        compDesc->set("sparsity_strategy", sparsityStrategyElements);
    }

    if (enableSaveStrategyToJsonFile)
    {
        log(Logger::MessageType::Info, "GraphOptimizer: Saving Strategy to JSON file");
        std::ofstream jsonOutputFile ;
        jsonOutputFile.open(jsonOutFileName, std::ios::out );
        if (!(jsonOutputFile.is_open()))
            log(Logger::MessageType::Info, "GraphOptimizer: Could not open output file " + jsonOutFileName);

        auto currentTime= chrono::system_clock::to_time_t(chrono::system_clock::now());
        std::string timeStamp(ctime(&currentTime));
        if (!timeStamp.empty() && timeStamp[timeStamp.length()-1] == '\n')
            timeStamp.erase(timeStamp.length()-1);

        mv::Element SSA("Streaming strategies generated by mcmCompiler "+timeStamp);
        mv::Element CSA("Clustering strategies generated by mcmCompiler "+timeStamp);
        mv::Element LSA("Tensor placement strategies generated by mcmCompiler "+timeStamp);
        mv::Element SpSA("Sparsity strategies generated by mcmCompiler "+timeStamp);
        SSA.set("streaming_strategy",streamingStrategyElements);
        CSA.set("split_strategy",multiClusterStrategyElements);
        LSA.set("tensor_placement_override",locationStrategyElements);
        SpSA.set("sparsity_strategy",sparsityStrategyElements);
        auto jsonSStrategy = SSA.toJSON(true);
        auto jsonCStrategy = CSA.toJSON(true);
        auto jsonLStrategy = LSA.toJSON(true);
        auto jsonSpStrategy = SpSA.toJSON(true);
        jsonOutputFile << jsonSStrategy.stringifyPretty() << "," << std::endl;
        jsonOutputFile << jsonCStrategy.stringifyPretty() << "," << std::endl;
        jsonOutputFile << jsonLStrategy.stringifyPretty()  << "," << std::endl;
        jsonOutputFile << jsonSpStrategy.stringifyPretty() << std::endl;

        jsonOutputFile.close();
    }

    auto clusters = globalConfig_["totalClusters"].get<int>();
    bool singleCluster = false;
    if(clusters == 1) singleCluster = true;
    // attach optimal tensor location (CMX or DDR) attribute to tensor
    for(auto strategy : allStrategies)
    {
        auto spilling = strategy["spilling"].get<bool>();
        auto opName   = strategy["name"].get<string>();

        auto op = model_.getOp(opName);
        if(op->getOpType() == "Output")
            continue;

        auto outTensor = op->getOutputTensor(0);

        if(spilling)
            outTensor->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::DDR);
        else
            outTensor->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::CMX);

        log(Logger::MessageType::Info, "GraphOptimizer: Output tensor location (from tensor attribute) for node " + op->getName() + " is " + outTensor->get("Location").toString());
    }
}

void StrategyManager::recursiveDijkstra(mv::Data::OpListIterator opBegin)
{
    struct costEdgeIteratorComp
    {
        bool operator()(const MetaGraph::edge_list_iterator lhs, const MetaGraph::edge_list_iterator rhs) const
        {
            int x = get<2>(*lhs);
            int y = get<2>(*rhs);
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

    std::unordered_set<std::string> recursedNodes;
    MetaGraph metaGraph;

    recursiveCriticalPath(opBegin, recursedNodes, metaGraph);
    log(Logger::MessageType::Info, "GraphOptimizer created MetaGraph of strategies. Searching for optimal path.");

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

    //build edge cost map required for call to dijkstra
    map<typename MetaGraph::edge_list_iterator, double, costEdgeIteratorComp> edgeCostMap;
    for (auto it = metaGraph.edge_begin(); it != metaGraph.edge_end(); ++it){
        edgeCostMap.insert(std::pair<MetaGraph::edge_list_iterator, double>(it, get<0>(*it)));
    }

    //call dijkstra on metagraph for each source and sink combo, save best path
    double max = inf_;
    vector<MetaGraph::edge_list_iterator> finalCriticalPath;
    bool foundCriticalPath = false;
    for(auto source : sources){
        for(auto sink : sinks) {
            vector<MetaGraph::edge_list_iterator> criticalPathEdges = dijkstra<std::tuple<mv::Op&,StrategySet,int>,MetaGraphEdge,costNodeIteratorComp,costEdgeIteratorComp, double>(metaGraph,source,sink,edgeCostMap);
            double cost = 0;

            for (auto edge : criticalPathEdges){
                cost += get<0>(*edge);
            }
            if(cost < max){
                foundCriticalPath = true;
                finalCriticalPath = criticalPathEdges;
                max = cost;
            }
        }
    }
    if(!foundCriticalPath)
        throw LogicError(*this, "GraphOptimizer: Unable to find non-infinite path through the MetaGraph. No strategy created.");

    log(Logger::MessageType::Info, "GraphOptimizer: Found optimal path, saving strategies.");
    saveMetaStrategy(finalCriticalPath);
}

/*
    This function builds a metaGraph by finding linear sections between pivot nodes (nodes with more
    than input or ouput node). When a pivot node is encountered, build all linear OptimizationGraphs beginning with that node, until
    the next pivot node is encountered. An OptimizationGraph has multiple nodes for each node in the original model graph, one for
    each potential strategy for that node.

    To create costs for the metaGraph, each edge will store the cumulative cost of all linear sections between the two pivot nodes as well
    as the strategies of the nodes along those critical paths. The only nodes in the metaGraph are the pivot nodes of the original graph.
 */
void StrategyManager::recursiveCriticalPath(typename graph<mv::Op, mv::DataFlow>::node_list_iterator modelSource,
                                            std::unordered_set<std::string>& recursedNodes, MetaGraph& metaGraph){
    struct costEdgeIteratorComp2
    {
        bool operator()(const OptimizationGraph::edge_list_iterator lhs,
                        const OptimizationGraph::edge_list_iterator rhs) const
        {
            return get<1>(*lhs) < get<1>(*rhs);
        }
    };

    struct costNodeIteratorComp
    {
        bool operator()(const OptimizationGraph::node_list_iterator lhs, const OptimizationGraph::node_list_iterator rhs) const
        {
            int x = get<2>(*lhs) ;
            int y = get<2>(*rhs) ;
            return (x < y);
        }
    };

    vector<vector<MetaGraphEdge>> parallelLinearSections;
    vector<MetaGraphEdge> parallelLinearSection;
    vector<OptimizationGraph> allOptimizationGraphs;
    vector<OptimizationGraph::node_list_iterator> first_nodes, last_nodes;

    mv::graph<mv::Op, mv::DataFlow> g;
    vector<typename graph<mv::Op, mv::DataFlow>::node_list_iterator> next_modelSource;

    //BASE CASE - end of graph, or previously recursed on this source
    std::string opName = (*modelSource).getName();
    if(modelSource->leftmost_child() == g.node_end() || recursedNodes.find(opName) != recursedNodes.end() || (*modelSource).getOpType() == "ConstantInt"){
        return;
    }
    int childCtr = 0;
    int nodeCtr = 0;

    recursedNodes.insert(opName);
    //RECURSIVE CASE - iterate over the children of source build linear subgraphs, kick off recursion if another pivot node hit
    for(auto model_edge  = modelSource->leftmost_output(); model_edge !=  g.edge_end(); ++model_edge)
    {
        if((*(model_edge->sink())).getOpType() == "ConstantInt")
            continue;
        childCtr++;
         //Create the subgraph to pass to dijkstra, build iteratively until hit next pivot node
        OptimizationGraph optimizationGraph;
        vector<vector<StrategySet>> nodeStrategies;
        vector<OptimizationGraph::node_list_iterator> old_nodes, new_nodes;
        first_nodes.clear();
        last_nodes.clear();
        old_nodes.clear();

        map<typename OptimizationGraph::edge_list_iterator, double, costEdgeIteratorComp2> edgeCostMap;
        int opCtr = 0;
        int optionCtr = 0;

        //Add modelSource (start pivot) to the graph
        vector<StrategySet> nodeStrategy;
        opCtr++;

        generateStrategySetForLayer(*modelSource,nodeStrategy);
        if(nodeStrategy.empty()){
            throw LogicError(*this, "GraphOptimizer did not create any potential strategies for " + (*modelSource).getName());
        }
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*modelSource,strategy,optionCtr++)));
        }
        first_nodes = new_nodes;

        for(const auto oldNode : old_nodes)
        {
            for(const auto newNode : new_nodes)
            {
                double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,OptimizationGraphEdge(edgeCost,nodeCtr));
                edgeCostMap.insert(std::pair<OptimizationGraph::edge_list_iterator, double>(newEdge, edgeCost));
                nodeCtr++;
            }
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
            if(nodeStrategy.empty()){
                throw LogicError(*this, "GraphOptimizer did not create any potential strategies for " + (*model_child).getName());
            }
            new_nodes.clear();
            for(auto strategy : nodeStrategy)
            {
                new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*model_child,strategy,optionCtr++)));
            }
            for(const auto oldNode : old_nodes){
                for(const auto newNode : new_nodes)
                {
                    double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                    int edgeCostInt = edgeCost ;
                    auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,OptimizationGraphEdge(edgeCost,nodeCtr));
                    edgeCostMap.insert(std::pair<OptimizationGraph::edge_list_iterator, double>(newEdge, edgeCost));
                    nodeCtr++;
                }
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
        if(nodeStrategy.empty()){
            throw LogicError(*this, "GraphOptimizer did not create any potential strategies for " + (*model_child).getName());
        }
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*model_child,strategy,optionCtr++)));
        }
        for(const auto oldNode : old_nodes){
            for(const auto newNode : new_nodes)
            {
                double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,OptimizationGraphEdge(edgeCost,nodeCtr));
                edgeCostMap.insert(std::pair<OptimizationGraph::edge_list_iterator, double>(newEdge, edgeCost));
                nodeCtr++;
            }
        }
        old_nodes.swap(new_nodes);
        nodeStrategies.push_back(nodeStrategy);
        last_nodes = old_nodes;

/*
    Call dijkstra on each source/sink strategy combination in the linear OptimizationGraph found from the modelSource on this  pass
 */
        allOptimizationGraphs.push_back(optimizationGraph);
        costEdgeIteratorComp2 costEdgeCompare;
        auto sinkNodeIt = optimizationGraph.node_begin() ;
        for (int ii=0; ii<optimizationGraph.node_size()-1; ii++) ++sinkNodeIt;

        int metaGraphEdgeCtr = 0;
        for(auto startingNode : first_nodes){
            for( auto endingNode : last_nodes)
            {
                CriticalEdges criticalPathEdges = dijkstra<OptimizationGraphNode,OptimizationGraphEdge,costNodeIteratorComp,costEdgeIteratorComp2, double>(optimizationGraph,startingNode,endingNode,edgeCostMap);
                double cost = 0;
                vector<StrategySet> strategies;

                for(auto edge : criticalPathEdges){
                    cost += get<0>(*edge);
                    strategies.push_back(get<1>(*edge->sink()));
                }
                strategies.pop_back();
                MetaGraphEdge particulars = MetaGraphEdge(cost, strategies, metaGraphEdgeCtr);
                metaGraphEdgeCtr++;
                parallelLinearSection.push_back(particulars);
            }
        }
        parallelLinearSections.push_back(parallelLinearSection);
        parallelLinearSection.clear();

    //TODO add check here for whether we've hit the same pivot (needed for nested parallelism)
    //TODO recurse if we haven't all hit the same pivot (into the nested parallel branch)
        next_modelSource.push_back(model_child);
        //writeDot(optimizationGraph, true); //debug
        //cout << "Dot at " << (*model_child).getName() << " has " << optimizationGraph.node_size() << " nodes" << endl;
    }

    //If child counter is 1, we were in a linear section, just add the subgraph to the metaGraph
    if(childCtr == 1){
        vector<MetaGraph::node_list_iterator> sources, sinks;

        if(metaGraph.node_size() == 0 ){  //If this is the first branch, add sources to MetaGraph
            for(auto source : first_nodes){
                std::tuple<mv::Op&,StrategySet,int> modifiedSource(get<0>(*source), get<1>(*source), metaGraph.node_size());
                sources.push_back(metaGraph.node_insert(modifiedSource));
            }
        }
        else{
            for(auto source : first_nodes){ //should be able to find all the sources, since they were previously a sink
                bool found = false;
                for(auto iter = metaGraph.node_begin(); iter != metaGraph.node_end() && !found; ++iter){
                    if(get<0>(*iter).getName() == get<0>(*source).getName() && get<1>(*iter) == get<1>(*source)){
                        sources.push_back(iter);
                        found = true;
                    }
                }
                if(!found){
                    throw LogicError(*this, "GraphOptimizer unable to find MetaGraph source throw" + get<0>(*source).getName());
                }
            }
        }
        for(auto sink : last_nodes){
            std::tuple<mv::Op&,StrategySet,int> modifiedSink(get<0>(*sink), get<1>(*sink), metaGraph.node_size());
            sinks.push_back(metaGraph.node_insert(modifiedSink));
        }
        vector<MetaGraphEdge> linearSection = parallelLinearSections[0];
        bool foundNonInf = false;
        for(int source = 0; source < sources.size(); source++){
            for(int sink = 0; sink < sinks.size(); sink++){
                MetaGraphEdge edgeInfo = linearSection[source*sinks.size() + sink];
                double cost = get<0>(edgeInfo);
                vector<StrategySet> strategies = get<1>(edgeInfo);
                metaGraph.edge_insert(sources[source], sinks[sink], MetaGraphEdge(cost, strategies, metaGraph.edge_size()));
                if(cost < inf_){
                    foundNonInf = true;
                }
            }
        }
        if(!foundNonInf){
            throw LogicError(*this, "GraphOptimizer: Found only infinite paths in linear section starting at " + get<0>(*sources.front()).getName());
        }
    }
    //In a parallel section, Add the sum of costs all critical edges and their strategies to create a MetaGraph edge
   if(childCtr > 1){
        vector<MetaGraph::node_list_iterator> sources, sinks;

        if(metaGraph.node_size() == 0 ){ //If this is the first branch, add sources to MetaGraph
            for(auto source : first_nodes){
                MetaGraphNode modifiedSource(get<0>(*source), get<1>(*source), metaGraph.node_size());
                sources.push_back(metaGraph.node_insert(modifiedSource));
            }
        }
        else{
            for(auto source : first_nodes){ //Should be able to find all the sources, since they were previously a sink to some branch
                bool found = false;
                for(auto iter = metaGraph.node_begin(); iter != metaGraph.node_end() && !found; ++iter){
                    if(get<0>(*iter).getName() == get<0>(*source).getName() && get<1>(*iter) == get<1>(*source)){
                        sources.push_back(iter);
                        found = true;
                    }
                }
                if(!found){
                    throw LogicError(*this, "GraphOptimizer: Unable to find MetaGraph source " + get<0>(*source).getName());
                }
            }
        }
        for(auto sink : last_nodes){
            MetaGraphNode modifiedSink(get<0>(*sink), get<1>(*sink), metaGraph.node_size());
            sinks.push_back(metaGraph.node_insert(modifiedSink));
        }

        bool foundNonInf = false;
        for(int source = 0; source < sources.size(); source++){
            for(int sink = 0; sink < sinks.size(); sink++){
                double cost = 0;
                vector<StrategySet> strategies;
                for(int sectionIdx = 0; sectionIdx < parallelLinearSections.size(); sectionIdx++)
                {
                    MetaGraphEdge edgeInfo = parallelLinearSections[sectionIdx][source*sinks.size() + sink];
                    cost += get<0>(edgeInfo);
                    for(auto strategy :  get<1>(edgeInfo)){
                        strategies.push_back(strategy);
                    }
                }
                metaGraph.edge_insert(sources[source], sinks[sink], MetaGraphEdge(cost, strategies, metaGraph.edge_size()));
                if(cost < inf_){
                    foundNonInf = true;
                }
                strategies.clear();
            }
        }
        if(!foundNonInf){
            throw LogicError(*this, "GraphOptimizer: Found only infinite paths in parallel section starting at " + get<0>(*sources.front()).getName());
        }
    }
    for(auto source : next_modelSource)
    {
        std::string next_opName = (*source).getName();
        if((*source).getOpType() != "ConstantInt" && recursedNodes.find(next_opName) == recursedNodes.end()){
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
