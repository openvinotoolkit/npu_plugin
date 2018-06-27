#include "include/mcm/base/printable.hpp"

mv::Printable::~Printable()
{
    
}

void mv::Printable::replaceSub(string &input, const string &oldSub, const string &newSub)
{
    string::size_type pos = 0u;
    while((pos = input.find(oldSub, pos)) != string::npos)
    {
        input.replace(pos, oldSub.length(), newSub);
        pos += newSub.length();
    }
}

mv::string mv::Printable::toString(const Printable &obj)
{
    return obj.toString();
}

mv::string mv::Printable::toString(int_type value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(float_type value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(unsigned_type value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(byte_type value)
{
    return toString((unsigned_type)value);
}

mv::string mv::Printable::toString(dim_type value)
{
    return toString((unsigned_type)value);
}

mv::string mv::Printable::toString(bool value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(DType value)
{
    switch (value)
    {
        case DType::Float:
            return "float";

        default:
            return "unknown";
    }
}

mv::string mv::Printable::toString(Order value)
{
    switch (value)
    {
        case Order::LastDimMajor:
            return "LastDimMajor";

        default:
            return "unknown";
    }
}

mv::string mv::Printable::toString(const mv::dynamic_vector<float> &value)
{
    return "(" + toString((unsigned_type)value.size()) + ")";
}

mv::string mv::Printable::toString(AttrType value)
{

    switch (value)
    {

        case AttrType::ByteType:
            return "byte";

        case AttrType::UnsingedType:
            return "unsigned";

        case AttrType::IntegerType:
            return "int";

        case AttrType::FloatType:
            return "float";
        
        case AttrType::DTypeType:
            return "dType";

        case AttrType::OrderType:
            return "order";

        case AttrType::ShapeType:
            return "shape";

        case AttrType::StringType:
            return "string";

        case AttrType::BoolType:
            return "bool";

        case AttrType::OpTypeType:
            return "opType";

        case AttrType::FloatVec2DType:
            return "floatVec2D";

        case AttrType::FloatVec3DType:
            return "floatVec3D";

        case AttrType::FloatVec4DType:
            return "floatVec4D";

        case AttrType::IntVec2DType:
            return "intVec2D";

        case AttrType::IntVec3DType:
            return "intVec3D";

        case AttrType::IntVec4DType:
            return "intVec4D";

        case AttrType::UnsignedVec2DType:
            return "unsignedVec2D";

        case AttrType::UnsignedVec3DType:
            return "unsignedVec3D";

        case AttrType::UnsignedVec4DType:
            return "unsignedVec4D";

        case AttrType::FloatVecType:
            return "floatVec";

        default:
            return "unknown";

    }
    
}

mv::string mv::Printable::toString(OpType value)
{

    switch (value)
    {
        case OpType::Input:
            return "input";

        case OpType::Output:
            return "output";

        case OpType::Constant:
            return "constant";

        case OpType::Conv2D:
            return "conv2D";

        case OpType::MatMul:
            return "matMul";

        case OpType::MaxPool2D:
            return "maxpool2D";

        case OpType::AvgPool2D:
            return "avgpool2D";

        case OpType::Concat:
            return "concat";
        
        case OpType::ReLU:
            return "relu";

        case OpType::Softmax:
            return "softmax";

        case OpType::Scale:
            return "scale";

        case OpType::BatchNorm:
            return "batchnorm";

        case OpType::Add:
            return "add";

        case OpType::Subtract:
            return "subtract";

        case OpType::Multiply:
            return "multiply";

        case OpType::Divide:
            return "divide";

        case OpType::Reshape:
            return "reshape";

        case OpType::Bias:
            return "bias";

        case OpType::FullyConnected:
            return "fullyConnected";

        default:
            return "unknown";

    }

}