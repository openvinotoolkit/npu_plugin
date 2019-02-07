#ifndef MV_TENSOR_DTYPE_HPP_
#define MV_TENSOR_DTYPE_HPP_

#include <string>
#include <unordered_map>
#include <functional>
#include "include/mcm/base/exception/dtype_error.hpp"
#include "include/mcm/tensor/binary_data.hpp"

namespace mv
{
    class DType : public LogSender
    {

    private:
        std::string dType_;

    public:

        DType();
        DType(const DType& other);
        DType(const std::string& value);

        std::string toString() const;

        DType& operator=(const DType& other);
        bool operator==(const DType& other) const;
        bool operator!=(const DType& other) const;

        std::string getLogID() const override;
        BinaryData toBinary(const std::vector<double>& data) const;
        unsigned getSizeInBits() const;
    };

}

#endif // MV_TENSOR_DTYPE_HPP_
