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

std::atomic<int> MetaEdge::unique_ctr(0);
std::atomic<int> MetaGraph::unique_ctr(0);
std::atomic<int> StrategyManager::unique_ctr(0);

StrategyManager::StrategyManager(OpModel& model,mv::Element& passDesc) :
        model_(model),passDesc_(passDesc)
{

}

//TODO:: error if the strategy is not there...
Attribute& StrategyManager::getStrategy(mv::Op op,string strategy)
{
    auto op_name = op.getName();
    auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");
    if (!(op.hasTypeTrait("optimizable")) || software)
    {
        log(Logger::MessageType::Debug, "StrategyManager: using Default strategy for " + op_name + " op");
        op_name = "Default";
    }
    auto layerEntry = layerStrategies_.find(op_name);

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

    return strategyEntry->second;
}

void  StrategyManager::setGlobalStrategy(string& name, Attribute& strategy)
{
    globalStrategies_[name]= strategy;
}

void StrategyManager::setGlobalConfig(string& name,Attribute& config)
{
    globalConfig_[name] = config;
}

const Attribute& StrategyManager::getGlobalConfig(const string& name) const
{
    auto it = globalConfig_.find(name);
    if(it == globalConfig_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return it->second;
}

const Attribute& StrategyManager::getGlobalStrategy(const string& name) const
{
    auto it = globalStrategies_.find(name);
    if(it == globalStrategies_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return it->second;
}

const StrategyManager::StrategySet& StrategyManager::getLayerStrategySet(const string& name) const
{
    auto it = layerStrategies_.find(name);
    if(it == layerStrategies_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return it->second;
}

bool StrategyManager::hasAttr(const GlobalSetting& map,const string& name) const
{
    return map.find(name) != map.end();
}

bool StrategyManager::hasAttr(const LayerStrategySet& map,const string& name) const
{
    return map.find(name) != map.end();
}

std::string StrategyManager::getLogID() const
{
    return "GraphOptimizer-StrategyManager";
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

std::vector<mv::Element> StrategyManager::convertStreamingStrategyToElement(CriticalPathNodes &strategiesToConvert, std::shared_ptr<mv::Element> compDesc)
{

    auto streamingStrategyList = compDesc->get<std::vector<mv::Element>>("streaming_strategy");

    //determine if node already has streaming strategy from JSON text, do not override text specification
    std::vector<std::string> hasSpec;
    for (auto s : streamingStrategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (unsigned i = 0; i < splitList.size(); i++)
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
    for (auto elem : strategiesToConvert)
    {
        auto& strategy = *elem;
        mv::Shape newStrategy = strategy["streaming"];
        std::string newName = strategy["name"] ;
        if ( std::find(hasSpec.begin(), hasSpec.end(), newName) == hasSpec.end())
        {
            copyElement.set("name_filter",newName);
            copySplits[0].set<int>("W", newStrategy[0]);
            copySplits[1].set<int>("H", newStrategy[1]);
            copySplits[2].set<int>("C", newStrategy[2]);
            copySplits[3].set<int>("K", newStrategy[3]);
            copyElement.set("splits",copySplits);
            streamingStrategyList.push_back(copyElement);
        }
    }

    return streamingStrategyList;
}

std::vector<mv::Element> StrategyManager::convertClusteringStrategyToElement(CriticalPathNodes &strategiesToConvert,
                                                                                 std::shared_ptr<mv::Element> compDesc)
{
    auto clusteringStrategyList = compDesc->get<std::vector<mv::Element>>("split_strategy");

    //determine if node already has clustering strategy from JSON text, do not override text specification
    std::vector<std::string> hasClusterSpec;
    for (auto s : clusteringStrategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        std::string strategyName = s.get<std::string>("strategy");
        if ((strategyName=="SplitOverH") or
            (strategyName=="SplitOverK") or
            (strategyName=="SplitOverHOverlapped") or
            (strategyName=="HKSwitch"))
        {
            hasClusterSpec.push_back(nodeName);
        }
    }

    //save clustering strategy into compilation descriptor
    mv::Element copyCElement("");//= clusteringStrategyList[0];
    for (auto elem : strategiesToConvert)
    {
        auto& strategy = *elem;
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

std::vector<mv::Element> StrategyManager::convertLocationStrategyToElement(CriticalPathNodes &strategiesToConvert)
{
    mv::Element copyLElement("");
    std::vector<mv::Element> locationStrategyList;

    for(auto elem : strategiesToConvert)
    {
        auto& strategy = *elem;
        auto spilling = strategy["spilling"].get<bool>();
        auto opName   = strategy["name"].get<string>();

        std::string DDRLocation = "DDR";
        std::string CMXLocation = "CMX";
        
        //todo::don't search the whole model for this
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

std::vector<mv::Element> StrategyManager::convertSparsityStrategyToElement(CriticalPathNodes &strategiesToConvert){
    log(Logger::MessageType::Debug, "GraphOptimizer: Converting Sparsity Strategies to Element");

    mv::Element copyLElement("");
    std::vector<mv::Element> sparsityStrategyList;

    for(auto elem: strategiesToConvert)
    {
        auto& strategy = *elem;
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

void StrategyManager::saveMetaStrategy(CriticalPathNodes& criticalPathNodes)
{
    struct {
        bool operator() (OptimizationGraph::node_list_iterator a,OptimizationGraph::node_list_iterator b) const
        {
            auto left = *a;
            auto right = *b;

            return left["name"].get<string>().compare(right["name"].get<string>()) < 0;
        }
    }strategyNameComparator;

    sort(criticalPathNodes.begin(),criticalPathNodes.end(),strategyNameComparator);
    const bool enableSaveStrategyToDescriptor = true;
    const bool enableSaveStrategyToJsonFile = true;

    auto globalParams = model_.getGlobalConfigParams();

    std::vector<mv::Element> streamingStrategyElements = convertStreamingStrategyToElement(criticalPathNodes, globalParams);
    std::vector<mv::Element> multiClusterStrategyElements = convertClusteringStrategyToElement(criticalPathNodes, globalParams);
    std::vector<mv::Element> locationStrategyElements = convertLocationStrategyToElement(criticalPathNodes);
    std::vector<mv::Element> sparsityStrategyElements = convertSparsityStrategyToElement(criticalPathNodes);

    if (enableSaveStrategyToDescriptor)
    {
        log(Logger::MessageType::Debug, "GraphOptimizer: Saving Strategy to Compilation Descriptor");
        auto compDesc = model_.getGlobalConfigParams();
        compDesc->set("streaming_strategy", streamingStrategyElements);
        compDesc->set("split_strategy", multiClusterStrategyElements);
        compDesc->set("sparsity_strategy", sparsityStrategyElements);
    }

    if (enableSaveStrategyToJsonFile)
    {
        log(Logger::MessageType::Debug, "GraphOptimizer: Saving Strategy to JSON file");
        std::ofstream jsonOutputFile ;
        jsonOutputFile.open(jsonOutFileName, std::ios::out );
        if (!(jsonOutputFile.is_open()))
            log(Logger::MessageType::Debug, "GraphOptimizer: Could not open output file " + jsonOutFileName);

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

    // attach optimal tensor location (CMX or DDR) attribute to tensor
    for(auto elem : criticalPathNodes)
    {
        auto& strategy = *elem;
        auto spilling = strategy["spilling"].get<bool>();
        auto opName   = strategy["name"].get<string>();

        auto op = model_.getOp(opName);
        if(op->getOpType() == "Output")
            continue;

        auto outTensor = op->getOutputTensor(0);

        if(spilling)
            outTensor->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::DDR);
        else
            outTensor->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);

        log(Logger::MessageType::Debug, "GraphOptimizer: Output tensor location (from tensor attribute) for node " + op->getName() + " is " + outTensor->get("Location").toString());
    }
}

void StrategyManager::initLayerStrategySets()
{
    for(auto opIt = model_.opBegin(); opIt != model_.opEnd() ; ++ opIt)
    {
        const auto& opType = opIt->getOpType();
        //todo:: have a generic trait marker for "Constant" operations at opDef ( among other generic traits todo's)
        if ((opType != "Constant") and
            (opType != "ConstantInt") and
            (opType != "ConstantDataElement") and
            (opType != "WeightsTable") and
            (opType != "SparsityMap"))
        {
            auto nodeStrategies = make_shared<vector<StrategySet>>(0);
            generateStrategySetForLayer(*opIt,*nodeStrategies);

            opIt->set<shared_ptr<vector<StrategySet>>>("StrategySet",nodeStrategies);
        }
    }

    return;
}

bool StrategyManager::isLinearGraph(mv::Data::OpListIterator opBegin,
                                        mv::Data::OpListIterator opEnd,
                                        vector<mv::Data::OpListIterator> children)
{
    if(children.size() > 1)
        return false;

    auto modelEnd = model_.opEnd();

    for(mv::Data::OpDFSIterator dfsIterator(children[0]); dfsIterator != opEnd; ++dfsIterator)
    {
        //we will need to check for linearity between 2 nodes; If we get to OpEnd troughout the DFS iteration
        //then the subGraph builder did some urecoverable logic error
        if(dfsIterator == modelEnd)
        {
            mv::LogicError(*this,"Logic error: recursive graphOptimizer got pivots " +
                            opBegin->getName() + " to " + opEnd->getName());
        }

        //While iterating DFS, if we get multiple child ops, then we have a pivot node
        if(dfsIterator.childrenSize() > 1)
        {
            return false;
        }
    }

    return true;
}

mv::Data::OpListIterator StrategyManager::naiveLCA(mv::Data::OpListIterator nodeA,mv::Data::OpListIterator nodeB,
                                                    mv::Data::OpListIterator opEnd)
{
    std::set<mv::Op*> nodeAChildren;

    mv::Data::OpDFSIterator naiveA(nodeA);
    mv::Data::OpDFSIterator naiveB(nodeB);

    do
    {
        nodeAChildren.insert( &(*naiveA));
        ++naiveA;
    }while(naiveA != opEnd);

    do
    {
        if(nodeAChildren.find( &(*naiveB)) != nodeAChildren.end())
        {
            return naiveB;
        }
        ++naiveB;
    }while(naiveB != opEnd);

    return opEnd;
}

mv::Data::OpListIterator StrategyManager::naiveLCA(vector<mv::Data::OpListIterator> children,mv::Data::OpListIterator opEnd)
{
    auto candidate = naiveLCA(children[0],children[1],opEnd);

    for( int childIdx = 2; childIdx < children.size(); childIdx++)
        candidate = naiveLCA(candidate,children[childIdx],opEnd);

    return candidate;
}

shared_ptr<vector<StrategyManager::SubGraph>> StrategyManager::extractSubgraphs(mv::Data::OpListIterator opBegin,
                                                                                    mv::Data::OpListIterator opEnd,
                                                                                    vector<mv::Data::OpListIterator> children)
{
    auto sGraphs = make_shared<vector<SubGraph>>();
    auto travelingNode = opBegin;
    auto travelingChildren = children;

    while(travelingNode != opEnd)
    {
        if(travelingChildren.size() == 1)
        {
            mv::Data::OpDFSIterator it(travelingChildren[0]);
            for( ;(it.childrenSize() == 1) and (it != opEnd); ++it );

            sGraphs->push_back( SubGraph(travelingNode,it,{travelingChildren[0]}));

//            cout<<"Traveled linear section " << travelingNode->getName() << " -> " << it->getName() << endl;
            travelingNode = it;
            travelingChildren.clear();

            for(auto child = travelingNode.leftmostChild(); child != model_.opEnd(); ++child)
            {
                travelingChildren.push_back(child);
            }
        }
        else
        {
            auto lcsa = naiveLCA(travelingChildren,opEnd);
//            cout << "Found branching out section " << travelingNode->getName() << "->" << lcsa->getName() << endl;

//             once we have the LCSA (lowest common SINGLE ancestor), we need to check for exclusivity of the branches.
//             we will do this via DFS-ing each child branch of the branching node, with the ending contition being the lcsa.
//             if we found a dfs path that exclusive (i.e. only this path touches the nodes), then it means we have a "good" subgraph
//             if we found branches with non-exclusive nodes, then they in summary will compose a subGraph.
//             The "special" scenario will arise, when there are no exclusive branches. This needs to go to the "special handilng"
//             TODO:: for now assume all child branches are exclusive, and just add them. Need to implement check to
//                    see if a path trough a specific child is exclusive or not, and group them until they become exclusive
//             TODO:: implement special case handling. If we cannot group children until they become exclusive, then need
//                    start removing edges until they do

            for(auto child = travelingNode.leftmostChild(); child != model_.opEnd(); ++child)
                sGraphs->push_back( SubGraph(travelingNode,lcsa,{child}));

            travelingNode = lcsa;

            travelingChildren.clear();
            for(auto child = travelingNode.leftmostChild(); child != model_.opEnd(); ++child)
            {
                travelingChildren.push_back(child);
            }
        }
    }
    return sGraphs;
}

std::shared_ptr<MetaGraph> StrategyManager::linearGraphSolver(mv::Data::OpDFSIterator opBegin,
                                                                mv::Data::OpDFSIterator opEnd,
                                                                mv::Data::OpDFSIterator firstChild)
{
//    cout << "Solving Linear Section " << opBegin->getName() << " -> " << opEnd->getName() << " via " << firstChild->getName() << endl;
    auto linearMeta = make_shared<MetaGraph>();
    auto modelEnd = model_.opEnd();

    auto cost = [this](Op& parentOp,Op& childOp,StrategySet& a,StrategySet& b) ->double
            {return this->transitionCost(parentOp,childOp,a,b); };

    //do first pivot out of the loop
    {
        auto nodeStrategies = opBegin->get<shared_ptr<vector<StrategySet>>>("StrategySet");
        linearMeta->addNewLevel(*opBegin,nodeStrategies,cost);
    }

    for(auto dfsIterator = firstChild; dfsIterator != opEnd; ++dfsIterator)
    {
        //we will need to check for linearity between 2 nodes; If we get to OpEnd troughout the DFS iteration
        //then the subGraph builder did some urecoverable logic error
        if(dfsIterator == modelEnd)
        {
            mv::LogicError(*this,"Logic error: recursive graphOptimizer got pivots " +
                            opBegin->getName() + " to " + opEnd->getName());
        }

        auto nodeStrategies = dfsIterator->get<shared_ptr<vector<StrategySet>>>("StrategySet");
        linearMeta->addNewLevel(*dfsIterator,nodeStrategies,cost);
    }

    //do last pivot out of the loop
    {
        auto nodeStrategies = opEnd->get<shared_ptr<vector<StrategySet>>>("StrategySet");
        linearMeta->addNewLevel(*opEnd,nodeStrategies,cost);
    }

    linearMeta->solve();

    return linearMeta;
}

std::shared_ptr<MetaGraph> StrategyManager::recursiveGraphSolver(mv::Data::OpListIterator opBegin,
                                                                    mv::Data::OpListIterator opEnd,
                                                                    vector<mv::Data::OpListIterator> children)
{
    if(isLinearGraph(opBegin,opEnd,children))
    {
        return linearGraphSolver(opBegin,opEnd,children[0]);
    }
    else
    {
        auto subGraphs = extractSubgraphs(opBegin,opEnd,children);

        vector<std::shared_ptr<MetaGraph>> childMetas;
        auto masterMeta = make_shared<MetaGraph>();

        for( auto sGraph : *(subGraphs.get()) )
        {
            auto& sGraphStart = get<0>(sGraph);
            auto& sGraphEnd   = get<1>(sGraph);
            auto& sGraphChildren = get<2>(sGraph);

            auto meta = recursiveGraphSolver(sGraphStart,sGraphEnd,sGraphChildren);
            childMetas.push_back(meta);
        }

        for(const auto& meta : childMetas)
        {
            if(createStrategyDots)
                meta->write(dotFileLocation,true);
            masterMeta->fuseMeta(meta);
        }

        masterMeta->solve();
        //todo:: implement sanity check function, to verify the metaGraph

        return masterMeta;
    }
}

void StrategyManager::graphParameterOptimizations()
{
    initLayerStrategySets();

    auto startingNode = model_.getInput();
    auto endingNode = model_.getOutput();
    vector<mv::Data::OpListIterator> children;

    for( auto child = startingNode.leftmostChild(); child != model_.opEnd(); ++child)
        children.push_back(child);

    auto generalizedLinearMeta = recursiveGraphSolver(startingNode,endingNode,children);
    if(createStrategyDots)
        generalizedLinearMeta->write(dotFileLocation,true);

    auto finalMetaGraph = make_shared<MetaGraph>();
    finalMetaGraph->fuseMeta(generalizedLinearMeta);
    finalMetaGraph->solve();

    auto bestPath = finalMetaGraph->getLowestCriticalPathExtended();
    saveMetaStrategy(*bestPath->nodes);

    if(createStrategyDots)
        finalMetaGraph->write(dotFileLocation,true);
}

void StrategyManager::generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec)
{
    //TODO:: error
    cout<<"ERROR generateStrategySetForLayer" << endl;
    return;
}

double StrategyManager::transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
{
    cout<<"ERROR transitionCost" << endl;
    return -1;
}
}
}
