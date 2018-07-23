#include "include/mcm/base/jsonable.hpp"

mv::Jsonable::~Jsonable()
{
    
}

mv::json::Value mv::Jsonable::toJsonValue(const Jsonable &obj)
{
    return obj.toJsonValue();
}

mv::json::Value mv::Jsonable::toJsonValue(int_type value)
{
    return mv::json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(float_type value)
{
    return mv::json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(unsigned_type value)
{
    return mv::json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(byte_type value)
{
    return mv::json::Value((unsigned_type)value);
}

mv::json::Value mv::Jsonable::toJsonValue(dim_type value)
{
    return mv::json::Value((unsigned_type)value);
}

mv::json::Value mv::Jsonable::toJsonValue(bool value)
{
    return mv::json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(DType value)
{
    switch (value)
    {
        case DType::Float:
            return mv::json::Value("float");

        default:
            return mv::json::Value("unknown");
    }
}

mv::json::Value mv::Jsonable::toJsonValue(Order value)
{
    switch (value)
    {
        case Order::LastDimMajor:
            return mv::json::Value(string("LastDimMajor"));

        default:
            return mv::json::Value(string("unknown"));
    }
}

mv::json::Value mv::Jsonable::toJsonValue(const mv::dynamic_vector<float> &value)
{
    return toJsonValue((unsigned_type)value.size());
}

mv::json::Value mv::Jsonable::toJsonValue(AttrType value)
{

    switch (value)
    {

        case AttrType::ByteType:
            return mv::json::Value("byte");

        case AttrType::UnsingedType:
            return mv::json::Value("unsigned");

        case AttrType::IntegerType:
            return mv::json::Value("int");

        case AttrType::FloatType:
            return mv::json::Value("float");
        
        case AttrType::DTypeType:
            return mv::json::Value("dType");

        case AttrType::OrderType:
            return mv::json::Value("order");

        case AttrType::ShapeType:
            return mv::json::Value("shape");

        case AttrType::StringType:
            return mv::json::Value("string");

        case AttrType::BoolType:
            return mv::json::Value("bool");

        case AttrType::OpTypeType:
            return mv::json::Value("opType");

        case AttrType::FloatVec2DType:
            return mv::json::Value("floatVec2D");

        case AttrType::FloatVec3DType:
            return mv::json::Value("floatVec3D");

        case AttrType::FloatVec4DType:
            return mv::json::Value("floatVec4D");

        case AttrType::IntVec2DType:
            return mv::json::Value("intVec2D");

        case AttrType::IntVec3DType:
            return mv::json::Value("intVec3D");

        case AttrType::IntVec4DType:
            return mv::json::Value("intVec4D");

        case AttrType::UnsignedVec2DType:
            return mv::json::Value("unsignedVec2D");

        case AttrType::UnsignedVec3DType:
            return mv::json::Value("unsignedVec3D");

        case AttrType::UnsignedVec4DType:
            return mv::json::Value("unsignedVec4D");

        case AttrType::FloatVecType:
            return mv::json::Value("floatVec");

        default:
            return mv::json::Value("unknown");

    }
    
}

mv::json::Value mv::Jsonable::toJsonValue(OpType value)
{

    switch (value)
    {
        case OpType::Input:
            return mv::json::Value("input");

        case OpType::Output:
            return mv::json::Value("output");

        case OpType::Constant:
            return mv::json::Value("constant");

        case OpType::Conv2D:
            return mv::json::Value("conv2D");

        case OpType::MatMul:
            return mv::json::Value("matMul");

        case OpType::MaxPool2D:
            return mv::json::Value("maxpool2D");

        case OpType::AvgPool2D:
            return mv::json::Value("avgpool2D");

        case OpType::Concat:
            return mv::json::Value("concat");
        
        case OpType::ReLU:
            return mv::json::Value("relu");

        case OpType::Softmax:
            return mv::json::Value("softmax");

        case OpType::Scale:
            return mv::json::Value("scale");

        case OpType::BatchNorm:
            return mv::json::Value("batchnorm");

        case OpType::Add:
            return mv::json::Value("add");

        case OpType::Subtract:
            return mv::json::Value("subtract");

        case OpType::Multiply:
            return mv::json::Value("multiply");

        case OpType::Divide:
            return mv::json::Value("divide");

        case OpType::Reshape:
            return mv::json::Value("reshape");

        case OpType::Bias:
            return mv::json::Value("bias");

        case OpType::FullyConnected:
            return mv::json::Value("fullyConnected");

        default:
            return mv::json::Value("unknown");

    }
}

mv::json::Value mv::Jsonable::toJsonValue(const string& value)
{
    return mv::json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(const char *value)
{
    return mv::json::Value(string(value));
}

