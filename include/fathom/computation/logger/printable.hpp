#ifndef PRINTABLE_HPP_
#define PRINTABLE_HPP_

#include <string>
#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    class Printable
    {

    public:

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

                case AttrType::TensorType:
                    return "const tensor";
                
                case AttrType::DTypeType:
                    return "dType";

                case AttrType::OrderType:
                    return "order";

                case AttrType::ShapeType:
                    return "shape";

                case AttrType::StringType:
                    return "string";

                default:
                    return "unknown";

            }
            
        }

    };

}

#endif
