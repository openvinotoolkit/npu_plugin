#include "include/mcm/tensor/dtype/dtype_registry.hpp"


namespace mv
{
    MV_DEFINE_REGISTRY(DTypeRegistry, std::string, mv::DTypeEntry)
}

mv::DTypeRegistry& mv::DTypeRegistry::instance()
{
    return Registry<DTypeRegistry, std::string, mv::DTypeEntry>::instance();
}
