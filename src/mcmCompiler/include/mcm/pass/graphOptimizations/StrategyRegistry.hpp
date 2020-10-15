#ifndef __STRATEGY_REGISTRY_HPP__
#define __STRATEGY_REGISTRY_HPP__

#include "include/mcm/base/registry.hpp"

#include "StrategyConfig.hpp"

#define EVAL(x) x

namespace mv {
namespace graphOptimizer {

class GlobalConfigRegistry : public Registry<GlobalConfigRegistry,std::string,AttributeEntry>
{
public:
    GlobalConfigRegistry() {};
    static GlobalConfigRegistry& instance();

#define MV_OPTIMIZER_REGISTER_GLOBAL_CONFIG(Name) \
    KEEP_SYMBOL(AttributeEntry& __MCM_REGISTER__ ## AttributeEntry ## __ ## Name) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance().enter(STRV(Name))

#define MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY(Sym) \
    KEEP_SYMBOL(AttributeEntry& __MCM_REGISTER__ ## AttributeEntry ## __ ## Sym) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance()
};

class GlobalStrategyRegistry : public Registry<GlobalStrategyRegistry,std::string,AttributeEntry>
{
public:
    GlobalStrategyRegistry() {};
    static GlobalStrategyRegistry& instance();

#define MV_OPTIMIZER_REGISTER_GLOBAL_STRATEGY(Name) \
    KEEP_SYMBOL(AttributeEntry& __MCM_REGISTER__ ## AttributeEntry ## __ ## Name) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance().enter(STRV(Name))

#define MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY(Sym) \
    KEEP_SYMBOL(AttributeEntry& __MCM_REGISTER__ ## AttributeEntry ## __ ## Sym) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance()
};

class LayerStrategyRegistry : public Registry<LayerStrategyRegistry,std::string,StrategySetEntry>
{
public:
    LayerStrategyRegistry() {};
    static LayerStrategyRegistry& instance();

#define MV_OPTIMIZER_REGISTER_LAYER_STRATEGY(Name) \
    KEEP_SYMBOL(StrategySetEntry& __MCM_REGISTER__ ## StrategySetEntry ## __ ## Name) =    \
    mv::graphOptimizer::LayerStrategyRegistry::instance().enter(STRV(Name))

#define MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(Sym) \
    KEEP_SYMBOL(StrategySetEntry& __MCM_REGISTER__ ## StrategySetEntry ## __ ## Sym) =    \
    mv::graphOptimizer::LayerStrategyRegistry::instance()
};
}
}


#endif //__STRATEGY_REGISTRY_HPP__
