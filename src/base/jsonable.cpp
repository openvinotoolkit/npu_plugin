#include "include/mcm/base/jsonable.hpp"

mv::Jsonable::~Jsonable()
{
    
}

mv::json::Value mv::Jsonable::toJsonValue(const Jsonable &obj)
{
    return obj.toJsonValue();
}

mv::json::Value mv::Jsonable::toJsonValue(int value)
{
    return json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(double value)
{
    return json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(unsigned value)
{
    return mv::json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(std::size_t value)
{
    return json::Value(static_cast<unsigned>(value));
}

mv::json::Value mv::Jsonable::toJsonValue(bool value)
{
    return json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(DType value)
{
    return mv::json::Value(mv::dtypeStrings.at(value));
}

mv::json::Value mv::Jsonable::toJsonValue(Order value)
{
    return mv::json::Value(mv::orderStrings.at(value));
}

mv::json::Value mv::Jsonable::toJsonValue(AttrType value)
{
    return mv::json::Value(mv::attrTypeStrings.at(value));
}

mv::json::Value mv::Jsonable::toJsonValue(OpType value)
{
    return mv::json::Value(mv::opsStrings.at(value));
}

mv::json::Value mv::Jsonable::toJsonValue(const std::string& value)
{
    return mv::json::Value(value);
}

mv::json::Value mv::Jsonable::toJsonValue(const char *value)
{
    return mv::json::Value(std::string(value));
}


//-----------------Base types constructor---------------------

int mv::Jsonable::constructIntTypeFromJson(json::Value& v)
{
    return v.get<long long>();
}

unsigned mv::Jsonable::constructUnsignedTypeFromJson(json::Value& v)
{
    return v.get<long long>();
}

double mv::Jsonable::constructFloatTypeFromJson(mv::json::Value& v)
{
    return v.get<double>();
}

std::size_t mv::Jsonable::constructByteTypeFromJson(mv::json::Value& v)
{
    return v.get<long long>();
}

bool mv::Jsonable::constructBoolTypeFromJson(mv::json::Value& v)
{
    return v.get<bool>();
}

mv::DType mv::Jsonable::constructDTypeFromJson(mv::json::Value& v)
{
    return mv::dtypeStringsReversed.at(v.get<std::string>());
}

mv::Order mv::Jsonable::constructOrderTypeFromJson(mv::json::Value& v)
{
    return mv::orderStringsReversed.at(v.get<std::string>());
}

mv::AttrType mv::Jsonable::constructAttrTypeFromJson(mv::json::Value& v)
{
    return mv::attrTypeStringsReversed.at(v.get<std::string>());
}

mv::OpType mv::Jsonable::constructOpTypeFromJson(mv::json::Value& v)
{
    return mv::opsStringsReversed.at(v.get<std::string>());
}

std::string mv::Jsonable::constructStringFromJson(mv::json::Value& v)
{
    return v.get<std::string>();
}

mv::FloatVector2D mv::Jsonable::constructFloatVector2DFromJson(mv::json::Value& v)
{
    mv::FloatVector2D vec;
    vec.e0 = constructFloatTypeFromJson(v[0]);
    vec.e1 = constructFloatTypeFromJson(v[1]);
    return vec;
}

mv::FloatVector3D mv::Jsonable::constructFloatVector3DFromJson(mv::json::Value& v)
{
    mv::FloatVector3D vec;
    vec.e0 = constructFloatTypeFromJson(v[0]);
    vec.e1 = constructFloatTypeFromJson(v[1]);
    vec.e2 = constructFloatTypeFromJson(v[2]);
    return vec;
}

mv::FloatVector4D mv::Jsonable::constructFloatVector4DFromJson(mv::json::Value& v)
{
    mv::FloatVector4D vec;
    vec.e0 = constructFloatTypeFromJson(v[0]);
    vec.e1 = constructFloatTypeFromJson(v[1]);
    vec.e2 = constructFloatTypeFromJson(v[2]);
    vec.e3 = constructFloatTypeFromJson(v[3]);
    return vec;
}

mv::IntVector2D mv::Jsonable::constructIntVector2DFromJson(mv::json::Value& v)
{
    mv::IntVector2D vec;
    vec.e0 = constructIntTypeFromJson(v[0]);
    vec.e1 = constructIntTypeFromJson(v[1]);
    return vec;
}

mv::IntVector3D mv::Jsonable::constructIntVector3DFromJson(mv::json::Value& v)
{
    mv::IntVector3D vec;
    vec.e0 = constructIntTypeFromJson(v[0]);
    vec.e1 = constructIntTypeFromJson(v[1]);
    vec.e2 = constructIntTypeFromJson(v[2]);
    return vec;
}

mv::IntVector4D mv::Jsonable::constructIntVector4DFromJson(mv::json::Value& v)
{
    mv::IntVector4D vec;
    vec.e0 = constructIntTypeFromJson(v[0]);
    vec.e1 = constructIntTypeFromJson(v[1]);
    vec.e2 = constructIntTypeFromJson(v[2]);
    vec.e3 = constructIntTypeFromJson(v[3]);
    return vec;
}

mv::UnsignedVector2D mv::Jsonable::constructUnsignedVector2DFromJson(mv::json::Value& v)
{
    mv::UnsignedVector2D vec;
    vec.e0 = constructUnsignedTypeFromJson(v[0]);
    vec.e1 = constructUnsignedTypeFromJson(v[1]);
    return vec;
}

mv::UnsignedVector3D mv::Jsonable::constructUnsignedVector3DFromJson(mv::json::Value& v)
{
    mv::UnsignedVector3D vec;
    vec.e0 = constructUnsignedTypeFromJson(v[0]);
    vec.e1 = constructUnsignedTypeFromJson(v[1]);
    vec.e2 = constructUnsignedTypeFromJson(v[2]);
    return vec;
}

mv::UnsignedVector4D mv::Jsonable::constructUnsignedVector4DFromJson(mv::json::Value& v)
{
    mv::UnsignedVector4D vec;
    vec.e0 = constructUnsignedTypeFromJson(v[0]);
    vec.e1 = constructUnsignedTypeFromJson(v[1]);
    vec.e2 = constructUnsignedTypeFromJson(v[2]);
    vec.e3 = constructUnsignedTypeFromJson(v[3]);
    return vec;
}

std::vector<double> mv::Jsonable::constructFloatVectorFromJson(mv::json::Value &v)
{
    std::vector<double> vec;
    for(unsigned i = 0; i < v.size(); ++i)
        vec.push_back(constructFloatTypeFromJson(v[i]));
    return vec;
}

std::vector<std::string> mv::Jsonable::constructStringVectorFromJson(mv::json::Value &v)
{
    std::vector<std::string> vec;
    for(unsigned i = 0; i < v.size(); ++i)
        vec.push_back(constructStringFromJson(v[i]));
    return vec;
}

