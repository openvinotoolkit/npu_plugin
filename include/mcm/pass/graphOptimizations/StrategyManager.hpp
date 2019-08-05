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

   /* using CriticalPath = std::tuple<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator,
            mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator,
            std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator>,
            double>;
*/ 

    using CriticalEdges = std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator>;
    using SharedCriticalEdges = std::vector<std::shared_ptr<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator>>;

    GlobalSetting globalConfig_;
    GlobalSetting globalStrategies_;
    LayerStrategySet layerStrategies_;

    OpModel& model_;
    mv::Element& passDesc_;

    string dotFileLocation;
/*  
    struct CriticalPath{
        //CriticalPath(){}
        CriticalPath(const CriticalPath &cp){ source = cp.source; sink = cp.sink; edges = cp.edges; sumCost = cp.sumCost;}
        CriticalPath(mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator so, mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator si, 
                std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator> e, double c)
                : source(so), sink(si), edges(std::move(e)), sumCost(c) {}

        mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator source;
        mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator sink;
        std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator> edges;
        double sumCost;
    };
*/

    StrategyManager(OpModel& model,mv::Element& passDesc);

    void updateValuesFromJSON();
    void updateDefaultValues();
    void printStrategy();

    void saveStrategy(std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator> cPathEdges);
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
