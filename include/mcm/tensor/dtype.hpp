#ifndef MV_TENSOR_DTYPE_HPP_
#define MV_TENSOR_DTYPE_HPP_

#include <string>
#include <unordered_map>
#include <functional>
#include "include/mcm/base/exception/dtype_error.hpp"
#include "include/mcm/tensor/binarydata.hpp"
#include "include/mcm/tensor/dtypetype.hpp"

namespace mv
{
    struct DTypeTypeHash
    {
        template <typename T>
        std::size_t operator()(T t) const
        {
            return static_cast<std::size_t>(t);
        }
    };

    class DType : public LogSender
    {

    private:

        static const std::unordered_map<DTypeType, std::string, DTypeTypeHash> dTypeStrings_;
        static const std::unordered_map<DTypeType, std::function<BinaryData(const std::vector<double>&)>,
                DTypeTypeHash> dTypeConvertors_;
        DTypeType dType_;

    public:

        DType();
        DType(DTypeType value);
        DType(const DType& other);
        DType(const std::string& value);

        std::string toString() const;

        DType& operator=(const DType& other);
        DType& operator=(const DTypeType& other);
        bool operator==(const DType& other) const;
        bool operator==(const DTypeType& other) const;
        bool operator!=(const DType& other) const;
        bool operator!=(const DTypeType& other) const;
        operator DTypeType() const;

        std::string getLogID() const override;
        BinaryData toBinary(const std::vector<double>& data) const;
    };

}

#endif // MV_TENSOR_DTYPE_HPP_
