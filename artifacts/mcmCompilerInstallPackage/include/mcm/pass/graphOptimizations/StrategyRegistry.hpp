#ifndef __STRATEGY_REGISTRY_HPP__
#define __STRATEGY_REGISTRY_HPP__

#include "include/mcm/base/registry.hpp"

#include "StrategyConfig.hpp"

namespace mv {
namespace graphOptimizer {

class GlobalConfigRegistry : public Registry<GlobalConfigRegistry,std::string,AttributeEntry>
{
public:
    GlobalConfigRegistry() {};
    static GlobalConfigRegistry& instance();

#define MV_OPTIMIZER_REGISTER_GLOBAL_CONFIG(Name) \
    static ATTRIBUTE_UNUSED(AttributeEntry& CONCATENATE(__ ## AttributeEntry ## __, __COUNTER__)) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance().enter(STRV(Name))

#define MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY() \
    static ATTRIBUTE_UNUSED(AttributeEntry& CONCATENATE(__ ## AttributeEntry ## __, __COUNTER__)) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance()
};

class GlobalStrategyRegistry : public Registry<GlobalStrategyRegistry,std::string,AttributeEntry>
{
public:
    GlobalStrategyRegistry() {};
    static GlobalStrategyRegistry& instance();

#define MV_OPTIMIZER_REGISTER_GLOBAL_STRATEGY(Name) \
    static ATTRIBUTE_UNUSED(AttributeEntry& CONCATENATE(__ ## AttributeEntry ## __, __COUNTER__)) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance().enter(STRV(Name))

#define MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY() \
    static ATTRIBUTE_UNUSED(AttributeEntry& CONCATENATE(__ ## AttributeEntry ## __, __COUNTER__)) =    \
    mv::graphOptimizer::GlobalConfigRegistry::instance()
};

class LayerStrategyRegistry : public Registry<LayerStrategyRegistry,std::string,StrategySetEntry>
{
public:
    LayerStrategyRegistry() {};
    static LayerStrategyRegistry& instance();

#define MV_OPTIMIZER_REGISTER_LAYER_STRATEGY(Name) \
    static ATTRIBUTE_UNUSED(StrategySetEntry& CONCATENATE(__ ## StrategySetEntry ## __, __COUNTER__)) =    \
    mv::graphOptimizer::LayerStrategyRegistry::instance().enter(STRV(Name))

#define MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY() \
    static ATTRIBUTE_UNUSED(StrategySetEntry& CONCATENATE(__ ## StrategySetEntry ## __, __COUNTER__)) =    \
    mv::graphOptimizer::LayerStrategyRegistry::instance()
};
}
}


#endif //__STRATEGY_REGISTRY_HPP__
