#ifndef JSONABLE_HPP
#define JSONABLE_HPP

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/op/ops_register.hpp"
#include "include/mcm/base/json/json.hpp"

namespace mv
{

    class Jsonable
    {

    public:

        virtual ~Jsonable() = 0;
        virtual json::Value toJsonValue() const = 0;

        static json::Value toJsonValue(const Jsonable &obj);
        static json::Value toJsonValue(int_type value);
        static json::Value toJsonValue(float_type value);
        static json::Value toJsonValue(unsigned_type value);
        static json::Value toJsonValue(byte_type value);
        static json::Value toJsonValue(dim_type value);
        static json::Value toJsonValue(bool value);
        static json::Value toJsonValue(DType value);
        static json::Value toJsonValue(Order value);
        static json::Value toJsonValue(const mv::dynamic_vector<float> &value);
        static json::Value toJsonValue(AttrType value);
        static json::Value toJsonValue(OpType value);
        static json::Value toJsonValue(const string &value);
        static json::Value toJsonValue(const char *value);

        template <class T>
        static json::Value toJsonValue(Vector2D<T> value)
        {
            json::Array arr;
            arr.append(Jsonable::toJsonValue(value.e0));
            arr.append(Jsonable::toJsonValue(value.e1));
            return json::Value(arr);
        }

        template <class T>
        static json::Value toJsonValue(Vector3D<T> value)
        {
            json::Array arr;
            arr.append(Jsonable::toJsonValue(value.e0));
            arr.append(Jsonable::toJsonValue(value.e1));
            arr.append(Jsonable::toJsonValue(value.e2));
            return json::Value(arr);
        }

        template <class T>
        static json::Value toJsonValue(Vector4D<T> value)
        {
            json::Array arr;
            arr.append(Jsonable::toJsonValue(value.e0));
            arr.append(Jsonable::toJsonValue(value.e1));
            arr.append(Jsonable::toJsonValue(value.e2));
            arr.append(Jsonable::toJsonValue(value.e3));

            return json::Value(arr);
        }

    };

}

#endif
