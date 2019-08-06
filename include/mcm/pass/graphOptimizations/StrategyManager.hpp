#ifndef __STRATEGY_MANAGER_HPP__
#define __STRATEGY_MANAGER_HPP__

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "math.h"
#include "tuple"

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

    
    using CriticalEdges = std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator>;
    using CriticalPair = std::pair<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>,CriticalEdges>;
    
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

    void saveStrategy(std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator> cPathEdges);
    void saveStrategyGraph(std::pair<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>,CriticalEdges> cPathEdges);
    void linearDijkstra(mv::Data::OpListIterator opBegin);
    void recursiveDijkstra(mv::Data::OpListIterator opBegin);
    std::vector<StrategyManager::CriticalEdges> recursiveCriticalPath
                        (typename graph<mv::Op, mv::DataFlow>::node_list_iterator modelSource, std::unordered_set<std::string>& recursedNodes);

    void writeDot(mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>& optimizationGraph,bool skipInf);

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
        return "GraphOptimizer-StrategyMangger";
    };

};


}
}

#endif //__STRATEGY_MANAGER_HPP__
