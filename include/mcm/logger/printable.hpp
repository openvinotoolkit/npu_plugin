#ifndef PRINTABLE_HPP_
#define PRINTABLE_HPP_

#include <string>
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/op/ops_register.hpp"

namespace mv
{

    class Printable
    {

    public:

        virtual ~Printable() = 0;
        static void replaceSub(string &input, const string &oldSub, const string &newSub);
        virtual string toString() const = 0;

        static string toString(const Printable &obj);
        static string toString(int_type value);
        static string toString(float_type value);
        static string toString(unsigned_type value);
        static string toString(byte_type value);
        static string toString(dim_type value);
        static string toString(bool value);
        static string toString(DType value);
        static string toString(Order value);
        static string toString(AttrType value);
        static string toString(OpType value);

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

    };

}

#endif
