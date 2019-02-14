#include "include/mcm/target/keembay/ppe_layer_type.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

const std::unordered_map<mv::PpeLayerTypeEnum, std::string, mv::PpeLayerTypeEnumHash> mv::PpeLayerType::ppeLayerTypeStrings_ =
{
    {PPELayerType_STORE, "STORE"},
    {PPELayerType_LOAD, "LOAD"},
    {PPELayerType_CLEAR, "CLEAR"},
    {PPELayerType_NOOP, "NOOP"},
    {PPELayerType_HALT, "HALT"},
    {PPELayerType_ADD, "ADD"},
    {PPELayerType_SUB, "SUB"},
    {PPELayerType_MULT, "MULT"},
    {PPELayerType_LRELU, "LRELU"},
    {PPELayerType_LRELUX, "LRELUX"},
    {PPELayerType_LPRELU, "LPRELU"},
    {PPELayerType_MAXIMUM, "MAXIMUM"},
    {PPELayerType_MINIMUM, "MINIMUM"},
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
    {PPELayerType_SIGMOID, "SIGMOID"},
    {PPELayerType_TANH, "TANH"},
    {PPELayerType_SQRT, "SQRT"},
    {PPELayerType_RSQRT, "RSQRT"},
    {PPELayerType_FLEXARB, "FLEXARB"}
};

mv::PpeLayerType::PpeLayerType(PpeLayerTypeEnum value) :
type_(value)
{

}

mv::PpeLayerType::PpeLayerType() :
type_(PpeLayerTypeEnum::PPELayerType_STORE)
{

}

mv::PpeLayerType::PpeLayerType(const PpeLayerType& other) :
type_(other.type_)
{

}

mv::PpeLayerType::PpeLayerType(const std::string& value)
{

    PpeLayerType(
        [=]()->PpeLayerType
        {
            for (auto &e : ppeLayerTypeStrings_)
                if (e.second == value)
                    return e.first;
            throw ArgumentError(*this, "Invalid initialization - string value specified as", value, "Initializer");
        }()
    );

}

std::string mv::PpeLayerType::toString() const
{
    return ppeLayerTypeStrings_.at(*this);
}

mv::PpeLayerType& mv::PpeLayerType::operator=(const PpeLayerType& other)
{
    type_ = other.type_;
    return *this;
}

mv::PpeLayerType& mv::PpeLayerType::operator=(const PpeLayerTypeEnum& other)
{
    type_ = other;
    return *this;
}

mv::PpeLayerType::operator PpeLayerTypeEnum() const
{
    return type_;
}

std::string mv::PpeLayerType::getLogID() const
{
    return "PPeLayerType:" + toString();
}
