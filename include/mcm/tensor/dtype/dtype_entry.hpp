#ifndef MV_DTYPE_ENTRY
#define MV_DTYPE_ENTRY

#include <vector>
#include <functional>
#include "include/mcm/tensor/binary_data.hpp"
#include "include/mcm/tensor/data_element.hpp"

namespace mv
{
    class DType;

    class DTypeEntry
    {

        std::string name_;
        std::function<BinaryData(const std::vector<mv::DataElement>&)> toBinaryFunc_;
        unsigned size_;
        bool isDoubleType_;

    public:

        DTypeEntry(const std::string& name);
        DTypeEntry& setToBinaryFunc(std::function<BinaryData(const std::vector<mv::DataElement>&)>& f);
        DTypeEntry& setSizeInBits(unsigned size);
        DTypeEntry& setIsDoubleType(bool isDouble);

        unsigned getSizeInBits() const;
        bool isDoubleType() const;

        const std::function<BinaryData(const std::vector<mv::DataElement>&)>& getToBinaryFunc() const;
    };

}

#endif // MV_DTYPE_ENTRY
