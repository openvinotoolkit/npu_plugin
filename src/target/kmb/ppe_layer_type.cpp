#include "include/mcm/target/kmb/ppe_layer_type.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

const std::unordered_map<mv::PPELayerTypeEnum, std::string, mv::PPELayerTypeEnumHash> mv::PPELayerType::ppeLayerTypeStrings_ =
{
    {PPELayerType_STORE, "STORE"},
    {PPELayerType_LOAD, "LOAD"},
    {PPELayerType_CLEAR, "CLEAR"},
    {PPELayerType_NOOP, "NOOP"},
    {PPELayerType_HALT, "HALT"},
    {PPELayerType_ADD, "Add"},
    {PPELayerType_SUB, "Subtract"},
    {PPELayerType_MULT, "Multiply"},
    {PPELayerType_RELU, "Relu"},
    {PPELayerType_RELUX, "RELUX"},
    {PPELayerType_LPRELU, "LeakyRelu"},
    {PPELayerType_MAXIMUM, "Maximum"},
    {PPELayerType_MINIMUM, "Minimum"},
    {PPELayerType_CEIL, "CEIL"},
    {PPELayerType_FLOOR, "FLOOR"},
    {PPELayerType_AND, "AND"},
    {PPELayerType_OR, "OR"},
    {PPELayerType_XOR, "XOR"},
    {PPELayerType_NOT, "NOT"},
    {PPELayerType_ABS, "ABS"},
    {PPELayerType_NEG, "NEG"},
    {PPELayerType_POW, "POW"},
    {PPELayerType_EXP, "EXP"},
    {PPELayerType_SIGMOID, "Sigmoid"},
    {PPELayerType_TANH, "TANH"},
    {PPELayerType_SQRT, "SQRT"},
    {PPELayerType_RSQRT, "RSQRT"},
    {PPELayerType_FLEXARB, "FLEXARB"}
};

mv::PPELayerType::PPELayerType(PPELayerTypeEnum value) :
type_(value)
{

}

mv::PPELayerType::PPELayerType() :
type_(PPELayerTypeEnum::PPELayerType_STORE)
{

}

mv::PPELayerType::PPELayerType(const PPELayerType& other) :
type_(other.type_)
{

}

mv::PPELayerType::PPELayerType(const std::string& value)
{
    auto enumFunctor = [=](const std::string& v)->PPELayerTypeEnum
    {
        for (auto &e : ppeLayerTypeStrings_)
            if (e.second == v)
                return e.first;
        throw ArgumentError(*this, "Invalid initialization - string value specified as", value, "Initializer");
    };
    auto correctEnum = enumFunctor(value);
    type_ = correctEnum;
}

std::string mv::PPELayerType::toString() const
{
    return ppeLayerTypeStrings_.at(*this);
}

mv::PPELayerType& mv::PPELayerType::operator=(const PPELayerType& other)
{
    type_ = other.type_;
    return *this;
}

mv::PPELayerType& mv::PPELayerType::operator=(const PPELayerTypeEnum& other)
{
    type_ = other;
    return *this;
}

mv::PPELayerType::operator PPELayerTypeEnum() const
{
    return type_;
}

std::string mv::PPELayerType::getLogID() const
{
    return "PPeLayerType:" + toString();
}
