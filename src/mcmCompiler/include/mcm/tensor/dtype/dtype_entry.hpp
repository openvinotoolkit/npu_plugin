#ifndef MV_DTYPE_ENTRY
#define MV_DTYPE_ENTRY

#include <vector>
#include <functional>
#include "include/mcm/tensor/data_element.hpp"

namespace mv
{
    class DTypeEntry
    {
        std::string name_;
        unsigned size_;
        bool isDoubleType_;
        bool isSigned_;
    public:

        DTypeEntry(const std::string& name);
        DTypeEntry& setSizeInBits(unsigned size);
        DTypeEntry& setIsDoubleType(bool isDouble);
        DTypeEntry& setIsSigned(bool isSigned);

        unsigned getSizeInBits() const;
        bool isDoubleType() const;
        bool isSigned() const;
    };

}

#endif // MV_DTYPE_ENTRY
