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
        static void replaceSub(std::string &input, const std::string &oldSub, const std::string &newSub);
        virtual std::string toString() const = 0;

        static std::string toString(const Printable &obj);
        static std::string toString(int value);
        static std::string toString(double value);
        static std::string toString(unsigned value);
        static std::string toString(unsigned long long value);
        static std::string toString(std::size_t value);
        static std::string toString(bool value);
        static std::string toString(DType value);
        static std::string toString(Order value);
        static std::string toString(const std::vector<double> &value);
        static std::string toString(const std::vector<std::string> &value);
        static std::string toString(AttrType value);
        static std::string toString(OpType value);

        template <class T>
        static std::string toString(Vector2D<T> value)
        {
            return "(" + Printable::toString(value.e0) + ", " + Printable::toString(value.e1) + ")"; 
        }

        template <class T>
        static std::string toString(Vector3D<T> value)
        {
            return "(" + Printable::toString(value.e0) + ", " + Printable::toString(value.e1) + ", " + Printable::toString(value.e2) + ")"; 
        }

        template <class T>
        static std::string toString(Vector4D<T> value)
        {
            return "(" + Printable::toString(value.e0) + ", " + Printable::toString(value.e1) + ", " + Printable::toString(value.e2) + ", " + Printable::toString(value.e3) + ")"; 
        }

    };

}

#endif
