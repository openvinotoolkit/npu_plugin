#ifndef MV_TENSOR_DTYPE_HPP_
#define MV_TENSOR_DTYPE_HPP_

#include <string>
#include <unordered_map>
#include <functional>
#include "include/mcm/base/exception/dtype_error.hpp"

namespace mv
{

    enum class DTypeType
    {
        Float16
    };

    struct BinaryData {
        DTypeType type_;

        union Data {
            std::vector<double>* fp64;
            std::vector<float>* fp32;
            std::vector<int16_t>* fp16;
            std::vector<uint8_t>* f8;
            std::vector<uint64_t>* u64;
            std::vector<uint32_t>* u32;
            std::vector<uint16_t>* u16;
            std::vector<uint8_t>* u8;
            std::vector<uint64_t>* i64;
            std::vector<int32_t>* i32;
            std::vector<int16_t>* i16;
            std::vector<int8_t>* i8;
            std::vector<int8_t>* i4;
            std::vector<int8_t>* i2;
            std::vector<int8_t>* i2x;
            std::vector<int8_t>* i4x;
            std::vector<int8_t>* bin;
            std::vector<int8_t>* log;
        } data_;

        BinaryData(DTypeType type) : type_(type)
        {

        }
        ~BinaryData()
        {
            switch(type_) {
                case DTypeType::Float16:
                    if (data_.fp16 != nullptr)
                        delete data_.fp16;
                    break;
                default:
                    break;
            }
        }
    };

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
