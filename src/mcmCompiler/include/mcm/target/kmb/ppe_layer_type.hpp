#ifndef MV_PPELAYERTYPE_HPP_
#define MV_PPELAYERTYPE_HPP_

#include <string>
#include <unordered_map>
#include "include/mcm/base/exception/argument_error.hpp"

namespace mv
{
    enum PPELayerTypeEnum
    {
        PPELayerType_STORE = 0,
        PPELayerType_LOAD = 1,
        PPELayerType_CLEAR = 2,
        PPELayerType_NOOP = 3,
        PPELayerType_HALT = 4,
        PPELayerType_ADD = 5,
        PPELayerType_SUB = 6,
        PPELayerType_MULT = 7,
        PPELayerType_RELU = 8,
        PPELayerType_RELUX = 9,
        PPELayerType_LPRELU = 10,
        PPELayerType_MAXIMUM = 11,
        PPELayerType_MINIMUM = 12,
        PPELayerType_CEIL = 13,
        PPELayerType_FLOOR = 14,
        PPELayerType_AND = 15,
        PPELayerType_OR = 16,
        PPELayerType_XOR = 17,
        PPELayerType_NOT = 18,
        PPELayerType_ABS = 19,
        PPELayerType_NEG = 20,
        PPELayerType_POW = 21,
        PPELayerType_EXP = 22,
        PPELayerType_SIGMOID = 23,
        PPELayerType_TANH = 24,
        PPELayerType_SQRT = 25,
        PPELayerType_RSQRT = 26,
        PPELayerType_FLEXARB = 27
    };

    struct PPELayerTypeEnumHash
    {
        template <typename T>
        std::size_t operator()(T t) const
        {
            return static_cast<std::size_t>(t);
        }
    };

    class PPELayerType : public LogSender
    {

    private:

        static const std::unordered_map<PPELayerTypeEnum, std::string, PPELayerTypeEnumHash> ppeLayerTypeStrings_;
        PPELayerTypeEnum type_;

    public:

        PPELayerType();
        PPELayerType(PPELayerTypeEnum value);
        PPELayerType(const PPELayerType& other);
        PPELayerType(const std::string& value);

        std::string toString() const;

        PPELayerType& operator=(const PPELayerType& other);
        PPELayerType& operator=(const PPELayerTypeEnum& other);
        operator PPELayerTypeEnum() const;

        std::string getLogID() const override;

        inline friend bool operator==(const PPELayerType& a, const PPELayerType& b)
        {
            return a.type_ == b.type_;
        }
        inline friend bool operator==(const PPELayerType& a, const PPELayerTypeEnum& b)
        {
            return a.type_ == b;
        }
        inline friend bool operator!=(const PPELayerType& a, const PPELayerType& b)
        {
            return a.type_ != b.type_;
        }
        inline friend bool operator!=(const PPELayerType& a, const PPELayerTypeEnum& b)
        {
            return a.type_ != b;
        }

    };

}

#endif
