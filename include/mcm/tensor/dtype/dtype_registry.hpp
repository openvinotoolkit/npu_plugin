#ifndef MV_DTYPE_REGISTRY_HPP_
#define MV_DTYPE_REGISTRY_HPP_

#include <string>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/tensor/dtype/dtype_entry.hpp"
#include "include/mcm/base/exception/dtype_error.hpp"
#include "include/mcm/base/exception/master_error.hpp"

namespace mv
{

    class DTypeRegistry : public Registry<DTypeRegistry, std::string, DTypeEntry>
    {


    public:

        static DTypeRegistry& instance();


        inline static bool checkDType(const std::string& dtype_string)
        {
            return instance().find(dtype_string) != nullptr;
        }

        static const std::function<BinaryData(const std::vector<double>&)>& getToBinaryFunc(const std::string& typeID)
        {

            if (!checkDType(typeID))
            {
                throw DTypeError("DTypeRegistry",
                        "Attempt of obtaining to-Binary conversion function for an unregistered dtype " + typeID);
            }

            mv::DTypeEntry* const typePtr = instance().find(typeID);

            if (typePtr)
            {
                return typePtr->getToBinaryFunc();
            }

            throw MasterError("DTypeRegistry", "Registered dtype " + typeID +
                " not found in the dtype registry");
        }

    };

    #define MV_REGISTER_DTYPE(Name)                          \
        MV_REGISTER_ENTRY(DTypeRegistry, std::string, DTypeEntry, #Name)    \

}

#endif // MV_DTYPE_REGISTRY_HPP_
