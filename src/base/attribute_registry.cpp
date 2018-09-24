#include "include/mcm/base/attribute_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(std::type_index, mv::attr::AttributeEntry)

}

mv::attr::AttributeRegistry& mv::attr::AttributeRegistry::instance()
{
    
    return static_cast<AttributeRegistry&>(Registry<std::type_index, AttributeEntry>::instance());

}