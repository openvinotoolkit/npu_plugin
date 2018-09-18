#include "include/mcm/computation/op/op_type.hpp"

constexpr unsigned short mv::OpType::Input;
constexpr unsigned short mv::OpType::Output;
constexpr unsigned short mv::OpType::Constant;
constexpr unsigned short mv::OpType::Conv2D;
constexpr unsigned short mv::OpType::Conversion;
constexpr unsigned short mv::OpType::MatMul;
constexpr unsigned short mv::OpType::MaxPool2D;
constexpr unsigned short mv::OpType::AvgPool2D;
constexpr unsigned short mv::OpType::Concat;
constexpr unsigned short mv::OpType::ReLU;
constexpr unsigned short mv::OpType::Softmax;
constexpr unsigned short mv::OpType::Scale;
constexpr unsigned short mv::OpType::BatchNorm;
constexpr unsigned short mv::OpType::Add;
constexpr unsigned short mv::OpType::Subtract;
constexpr unsigned short mv::OpType::Multiply;
constexpr unsigned short mv::OpType::Divide;
constexpr unsigned short mv::OpType::Reshape;
constexpr unsigned short mv::OpType::Bias;
constexpr unsigned short mv::OpType::FullyConnected;

const std::unordered_map<unsigned short, std::string> mv::OpType::opTypeStrings_ =
{
    {Input, "Input"},
    {Output, "Output"},
    {Constant, "Constant"},
    {Conv2D, "Conv2D"},
    {Conversion, "Conversion"},
    {MatMul, "MatMul"},
    {MaxPool2D, "MaxPool2D"},
    {AvgPool2D, "AvgPool2D"},
    {Concat, "Concat"},
    {ReLU, "ReLU"},
    {Softmax, "Softmax"},
    {Scale, "Scale"},
    {BatchNorm, "BatchNorm"},
    {Add, "Add"},
    {Subtract, "Subtract"},
    {Multiply, "Multiply"},
    {Divide, "Divide"},
    {Reshape, "Reshape"},
    {Bias, "Bias"},
    {FullyConnected, "FullyConnected"}
};

mv::OpType::OpType() :
opType_(0)
{

}

mv::OpType::OpType(const unsigned short value) :
opType_(value)
{

    if (opTypeStrings_.find(value) == opTypeStrings_.end())
        throw OpError(*this, "Invalid initialization - int value specified as " + std::to_string(value));
    
}

mv::OpType::OpType(const std::string& value)
{
    OpType(
        [=]()->unsigned short
        {
            for (auto &e : opTypeStrings_) 
                if (e.second == value) 
                    return e.first;
            throw OpError(*this, "Invalid initialization - string value specified as " + value);
        }()
    );
}

std::string mv::OpType::toString() const
{
    return opTypeStrings_.at(opType_);
}

bool mv::OpType::operator==(const OpType &other) const
{
    return opType_ == other.opType_;
}

bool mv::OpType::operator!=(const OpType &other) const
{
    return !operator==(other);
}

bool mv::OpType::operator==(unsigned short value) const
{
    OpType other(value);
    return operator==(other);
}

bool mv::OpType::operator!=(unsigned short value) const
{
    return !operator==(value);
}

bool mv::OpType::operator<(const OpType &other) const
{
    return opType_ < other.opType_;
}

mv::OpType::operator unsigned short() const
{
    return opType_;
}

std::string mv::OpType::getLogID() const
{
    return "OpType";
}