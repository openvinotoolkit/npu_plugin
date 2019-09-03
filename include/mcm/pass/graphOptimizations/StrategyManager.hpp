#ifndef __STRATEGY_MANAGER_HPP__
#define __STRATEGY_MANAGER_HPP__

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "math.h"
#include <unordered_set>
#include "tuple"
#include "limits"

namespace mv {
namespace graphOptimizer  {

using namespace std;
using namespace std::placeholders;

class StrategyManager : public LogSender
{
public:
    using GlobalSetting     = unordered_map<string,Attribute>;
    using StrategySet       = unordered_map<string,Attribute>;
    using LayerStrategySet  = unordered_map<string,StrategySet>;

    using OptimizationGraphNode = std::tuple<mv::Op&,StrategySet,int>; //op, strategies, unique id
    using OptimizationGraphEdge = std::pair<double,int>; //cost, unique id
    using OptimizationGraph = mv::graph<OptimizationGraphNode,OptimizationGraphEdge>;

    using MetaGraphNode = OptimizationGraphNode;
    using MetaGraphEdge = std::tuple<double, vector<StrategySet>, int>; //cost, strategies, unique id
    using MetaGraph = mv::graph<MetaGraphNode, MetaGraphEdge>;

    using CriticalEdges = std::vector<OptimizationGraph::edge_list_iterator>;

    
    static constexpr auto inf_ = numeric_limits<double>::infinity();

    GlobalSetting globalConfig_;
    GlobalSetting globalStrategies_;
    LayerStrategySet layerStrategies_;

    OpModel& model_;
    mv::Element& passDesc_;

    string dotFileLocation;

    StrategyManager(OpModel& model,mv::Element& passDesc);

    void updateValuesFromJSON();
    void updateDefaultValues();
    void printStrategy();

    std::vector<mv::Element> convertStreamingStrategyToElement(std::vector<StrategySet> &strategiesToConvert, std::shared_ptr<mv::Element> compDesc);
    std::vector<mv::Element> convertClusteringStrategyToElement(std::vector<StrategySet> &strategiesToConvert, std::shared_ptr<mv::Element> compDesc);
    std::vector<mv::Element> convertLocationStrategyToElement(std::vector<StrategySet> &strategiesToConvert);
    void saveStrategyToJsonFile(std::vector<mv::Element> &stategiesToSave,std::string jsonOutputFileName);
    void saveStrategyToCompilationDescriptor(vector<mv::Element> &stategiesToSave, std::shared_ptr<mv::Element> compDesc);
    void saveMetaStrategy(std::vector<MetaGraph::edge_list_iterator> cPathEdges);
    void recursiveDijkstra(mv::Data::OpListIterator opBegin);
    void recursiveCriticalPath(typename graph<mv::Op, mv::DataFlow>::node_list_iterator modelSource, std::unordered_set<std::string>& recursedNodes, MetaGraph& metaGraph);

    void writeDot(OptimizationGraph& graph,bool skipInf);
    void writeMetaDot(MetaGraph& graph, bool skipInf);
    string strategyString(OptimizationGraphNode n);

    virtual void generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec);
    virtual double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child);
    virtual ~StrategyManager() {};

    Attribute& getStrategy(mv::Op op,string strategy);

    inline void setGlobalConfig(string& name,Attribute& config)
    {
        globalConfig_[name] = config;
    }
    inline void  setGlobalStrategy(string& name, Attribute& strategy)
    {
        globalStrategies_[name]= strategy;
    }

    inline const Attribute& getGlobalConfig(const string& name) const
    {
        auto it = globalConfig_.find(name);
        if(it == globalConfig_.end())
            throw ArgumentError(*this, "name", name, "Undefined attribute");
        return it->second;
    }

    inline const Attribute& getGlobalStrategy(const string& name) const
    {
        auto it = globalStrategies_.find(name);
        if(it == globalStrategies_.end())
            throw ArgumentError(*this, "name", name, "Undefined attribute");
        return it->second;
    }

    inline const StrategySet& getLayerStrategySet(const string& name) const
    {
        auto it = layerStrategies_.find(name);
        if(it == layerStrategies_.end())
            throw ArgumentError(*this, "name", name, "Undefined attribute");
        return it->second;
    }

    inline bool hasAttr(const GlobalSetting& map,const string& name) const
    {
        return map.find(name) != map.end();
    }

    inline bool hasAttr(const LayerStrategySet& map,const string& name) const
    {
        return map.find(name) != map.end();
    }

    std::string getLogID() const
    {
        return "GraphOptimizer-StrategyManager";
    };

};


}
}

#endif //__STRATEGY_MANAGER_HPP__
