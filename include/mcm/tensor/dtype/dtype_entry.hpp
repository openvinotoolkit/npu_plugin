#ifndef MV_DTYPE_ENTRY
#define MV_DTYPE_ENTRY

#include <vector>
#include <functional>
#include "include/mcm/tensor/data_element.hpp"

namespace mv
{
    class DType;

    class DTypeEntry
    {
        std::string name_;
        unsigned size_;
        bool isDoubleType_;

    public:

        DTypeEntry(const std::string& name);
        DTypeEntry& setSizeInBits(unsigned size);
        DTypeEntry& setIsDoubleType(bool isDouble);

        unsigned getSizeInBits() const;
        bool isDoubleType() const;
    };

}

#endif // MV_DTYPE_ENTRY
