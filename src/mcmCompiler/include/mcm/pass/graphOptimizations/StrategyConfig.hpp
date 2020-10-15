#ifndef __STRATEGY_CONFIG_HPP__
#define __STRATEGY_CONFIG_HPP__

#include <map>
#include "include/mcm/op_model.hpp"

namespace mv {
namespace graphOptimizer {

using namespace std;

class AttributeEntry
{
    string config_;
    Attribute attr_;
public:

    AttributeEntry(string config)
    {
        config_ = config;
    }

    AttributeEntry& set(bool attr)
    {
        attr_ = attr;
        return *this;
    }
//
    AttributeEntry& set(int attr)
    {
        attr_ = attr;
        return *this;
    }

    AttributeEntry& set(string& attr)
    {
        attr_ = attr;
        return *this;
    }

    Attribute& getAttr()
    {
        return attr_;
    }

};

class StrategySetEntry
{
private:
    using StrategySet = unordered_map<string,Attribute>;
    string layer_;
    StrategySet layerStrategies_;
    string current_;

public:
    StrategySetEntry(string layer)
    {
        current_ = "";
        layer_ = layer;
    }

//    StrategySetEntry& insert(string setName,set<Attribute>& strategySet)
//    {
//        layerStrategies_[setName] = strategySet;
//        return *this;
//    }

    StrategySetEntry& registerSet(string setName)
    {
        current_ = setName;
        return *this;
    }

    StrategySetEntry& insert(Attribute strategy)
    {
        if( current_.compare("") != 0)
        {
            layerStrategies_[current_] = strategy;
        }
        return *this;
    }

    StrategySet& getStrategySet()
    {
        return layerStrategies_;
    }

};



}
}
#endif
