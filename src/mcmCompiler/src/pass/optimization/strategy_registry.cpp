#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"

namespace mv {

MV_DEFINE_REGISTRY(graphOptimizer::GlobalConfigRegistry, std::string, graphOptimizer::AttributeEntry)
MV_DEFINE_REGISTRY(graphOptimizer::GlobalStrategyRegistry, std::string, graphOptimizer::AttributeEntry)
MV_DEFINE_REGISTRY(graphOptimizer::LayerStrategyRegistry, std::string, graphOptimizer::StrategySetEntry)

namespace graphOptimizer {

GlobalConfigRegistry& GlobalConfigRegistry::instance()
{
    return Registry<GlobalConfigRegistry, std::string, AttributeEntry>::instance();
}

GlobalStrategyRegistry& GlobalStrategyRegistry::instance()
{
    return Registry<GlobalStrategyRegistry, std::string, AttributeEntry>::instance();
}

LayerStrategyRegistry& LayerStrategyRegistry::instance()
{
    return Registry<LayerStrategyRegistry, std::string, StrategySetEntry>::instance();
}

}
}
