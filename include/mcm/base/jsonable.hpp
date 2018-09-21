#ifndef JSONABLE_HPP
#define JSONABLE_HPP

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/op/ops_register.hpp"
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/computation/model/types.hpp"


namespace mv
{

    class Jsonable
    {

    public:

        virtual ~Jsonable() = 0;
        virtual json::Value toJsonValue() const = 0;

        //Base type -> Json value
        static json::Value toJsonValue(const Jsonable &obj);
        static json::Value toJsonValue(int_type value);
        static json::Value toJsonValue(float_type value);
        static json::Value toJsonValue(unsigned_type value);
        static json::Value toJsonValue(size_t value);
        static json::Value toJsonValue(byte_type value);
        static json::Value toJsonValue(dim_type value);
        static json::Value toJsonValue(bool value);
        static json::Value toJsonValue(DType value);
        static json::Value toJsonValue(Order value);
        static json::Value toJsonValue(AttrType value);
        static json::Value toJsonValue(OpType value);
        static json::Value toJsonValue(const string &value);
        static json::Value toJsonValue(const char *value);

        //Json value -> base type
        static int_type constructIntTypeFromJson(mv::json::Value& v);
        static unsigned_type constructUnsignedTypeFromJson(mv::json::Value& v);
        static size_t constructSizeTypeFromJson(mv::json::Value& v);
        static float_type constructFloatTypeFromJson(mv::json::Value& v);
        static byte_type constructByteTypeFromJson(mv::json::Value& v);
        static dim_type constructDimTypeFromJson(mv::json::Value& v);
        static bool constructBoolTypeFromJson(mv::json::Value& v);
        static DType constructDTypeFromJson(mv::json::Value& v);
        static Order constructOrderTypeFromJson(mv::json::Value& v);
        static AttrType constructAttrTypeFromJson(mv::json::Value& v);
        static OpType constructOpTypeFromJson(mv::json::Value& v);
        static string constructStringFromJson(mv::json::Value& v);
        static IntVector2D constructIntVector2DFromJson(mv::json::Value &v);
        static IntVector3D constructIntVector3DFromJson(mv::json::Value &v);
        static IntVector4D constructIntVector4DFromJson(mv::json::Value &v);
        static FloatVector2D constructFloatVector2DFromJson(mv::json::Value &v);
        static FloatVector3D constructFloatVector3DFromJson(mv::json::Value &v);
        static FloatVector4D constructFloatVector4DFromJson(mv::json::Value &v);
        static UnsignedVector2D constructUnsignedVector2DFromJson(mv::json::Value &v);
        static UnsignedVector3D constructUnsignedVector3DFromJson(mv::json::Value &v);
        static UnsignedVector4D constructUnsignedVector4DFromJson(mv::json::Value &v);
        static mv::dynamic_vector<float_type> constructFloatVectorFromJson(mv::json::Value &v);
        static mv::dynamic_vector<mv::string> constructStringVectorFromJson(mv::json::Value &v);
        static mv::dynamic_vector<unsigned> constructUnsignedVectorFromJson(mv::json::Value &v);
        static mv::dynamic_vector<size_t> constructSizeVectorFromJson(mv::json::Value &v);

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

        template <class T>
        static json::Value toJsonValue(mv::dynamic_vector<T> value)
        {
            mv::json::Array a;
            for(auto x = value.begin(); x != value.end(); ++x)
                a.append(mv::Jsonable::toJsonValue(*x));
            return mv::json::Value(a);
        }

    };

}

#endif
