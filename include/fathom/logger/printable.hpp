#ifndef PRINTABLE_HPP_
#define PRINTABLE_HPP_

#include <string>
#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/op/ops_register.hpp"

namespace mv
{

    class Printable
    {

    public:

        virtual ~Printable()
        {
            
        }

        inline static void replaceSub(string &input, const string &oldSub, const string &newSub)
        {
            string::size_type pos = 0u;
            while((pos = input.find(oldSub, pos)) != string::npos)
            {
                input.replace(pos, oldSub.length(), newSub);
                pos += newSub.length();
            }
        }

        virtual string toString() const = 0;

        static string toString(const Printable &obj)
        {
            return obj.toString();
        }

        static string toString(int_type value)
        {
            return std::to_string(value);
        }

        static string toString(float_type value)
        {
            return std::to_string(value);
        }

        static string toString(unsigned_type value)
        {
            return std::to_string(value);
        }

        static string toString(byte_type value)
        {
            return toString((unsigned_type)value);
        }

        static string toString(dim_type value)
        {
            return toString((unsigned_type)value);
        }

        static string toString(bool value)
        {
            return std::to_string(value);
        }

        static string toString(DType value)
        {
            switch (value)
            {
                case DType::Float:
                    return "float";

                default:
                    return "unknown";
            }
        }

        static string toString(Order value)
        {
            switch (value)
            {
                case Order::NWHC:
                    return "NWHC";

                default:
                    return "unknown";
            }
        }

        template <class T>
        static string toString(Vector2D<T> value)
        {
            return "(" + Printable::toString(value.e0) + ", " + Printable::toString(value.e1) + ")"; 
        }

        template <class T>
        static string toString(Vector3D<T> value)
        {
            return "(" + Printable::toString(value.e0) + ", " + Printable::toString(value.e1) + ", " + Printable::toString(value.e2) + ")"; 
        }

        template <class T>
        static string toString(Vector4D<T> value)
        {
            return "(" + Printable::toString(value.e0) + ", " + Printable::toString(value.e1) + ", " + Printable::toString(value.e2) + ", " + Printable::toString(value.e3) + ")"; 
        }

        static string toString(AttrType value)
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

                default:
                    return "unknown";

            }
            
        }

        static string toString(OpType value)
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

                case OpType::FullyConnected:
                    return "fullyConnected";

                case OpType::MaxPool2D:
                    return "maxpool2D";

                case OpType::AvgPool2D:
                    return "avgpool2D";

                case OpType::Concat:
                    return "concat";
                
                case OpType::ReLu:
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

                case OpType::Muliply:
                    return "multiply";

                case OpType::Divide:
                    return "divide";

                case OpType::Reshape:
                    return "reshape";

                default:
                    return "unknown";

            }

        }

    };

}

#endif
