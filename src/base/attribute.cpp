#include "include/mcm/base/attribute.hpp"

mv::allocator mv::Attribute::allocator_;

/*mv::byte_type mv::Attribute::nextTypeId()
{
    static byte_type id(0);
    assert(id < max_byte && "Out of attribute types ID");
    return id++;
}*/

mv::Attribute::Attribute() :
attrType_(AttrType::UnknownType)
{

}

mv::Attribute mv::Attribute::JsonAttributeFactory(mv::json::Value& value)
{
    string attributeType = constructStringFromJson(value["attrType"]);
    mv::AttrType attr = mv::attrTypeStringsReversed.at(attributeType);
    switch (attr)
    {
        case mv::AttrType::ByteType:
            return Attribute(mv::AttrType::ByteType, constructByteTypeFromJson(value["content"]));
        case mv::AttrType::BoolType:
            return Attribute(mv::AttrType::BoolType, constructBoolTypeFromJson(value["content"]));
        case mv::AttrType::UnsignedType:
            return Attribute(mv::AttrType::UnsignedType, constructUnsignedTypeFromJson(value["content"]));
        case mv::AttrType::IntegerType:
            return Attribute(mv::AttrType::IntegerType, constructIntTypeFromJson(value["content"]));
        case mv::AttrType::FloatType:
            return Attribute(mv::AttrType::FloatType, constructFloatTypeFromJson(value["content"]));
        case mv::AttrType::DTypeType:
            return Attribute(mv::AttrType::DTypeType, constructDTypeFromJson(value["content"]));
        case mv::AttrType::OrderType:
            return Attribute(mv::AttrType::OrderType, constructOrderTypeFromJson(value["content"]));
        case mv::AttrType::ShapeType:
            return Attribute(mv::AttrType::ShapeType, Shape(value["content"]));
        case mv::AttrType::StringType:
            return Attribute(mv::AttrType::StringType, constructStringFromJson(value["content"]));
        case mv::AttrType::OpTypeType:
            return Attribute(mv::AttrType::OpTypeType, constructOpTypeFromJson(value["content"]));
        case mv::AttrType::FloatVec2DType:
            return Attribute(mv::AttrType::FloatVec2DType, constructFloatVector2DFromJson(value["content"]));
        case mv::AttrType::FloatVec3DType:
            return Attribute(mv::AttrType::FloatVec3DType, constructFloatVector3DFromJson(value["content"]));
        case mv::AttrType::FloatVec4DType:
            return Attribute(mv::AttrType::FloatVec4DType, constructFloatVector4DFromJson(value["content"]));
        case mv::AttrType::IntVec2DType:
            return Attribute(mv::AttrType::IntVec2DType, constructIntVector2DFromJson(value["content"]));
        case mv::AttrType::IntVec3DType:
            return Attribute(mv::AttrType::IntVec3DType, constructIntVector3DFromJson(value["content"]));
        case mv::AttrType::IntVec4DType:
            return Attribute(mv::AttrType::IntVec4DType, constructIntVector4DFromJson(value["content"]));
        case mv::AttrType::UnsignedVec2DType:
            return Attribute(mv::AttrType::UnsignedVec2DType, constructUnsignedVector2DFromJson(value["content"]));
        case mv::AttrType::UnsignedVec3DType:
            return Attribute(mv::AttrType::UnsignedVec3DType, constructUnsignedVector3DFromJson(value["content"]));
        case mv::AttrType::UnsignedVec4DType:
            return Attribute(mv::AttrType::UnsignedVec4DType, constructUnsignedVector4DFromJson(value["content"]));
        case mv::AttrType::FloatVecType:
            return Attribute(mv::AttrType::FloatVecType, constructFloatVectorFromJson(value["content"]));
        case mv::AttrType::StringVecType:
            return Attribute(mv::AttrType::StringVecType, constructStringVectorFromJson(value["content"]));
        case mv::AttrType::UnsignedVecType:
            return Attribute(mv::AttrType::UnsignedVecType, constructUnsignedVectorFromJson(value["content"]));
        case mv::AttrType::UnknownType:
            return Attribute(mv::AttrType::UnknownType, nullptr);
    }
    return Attribute(mv::AttrType::UnknownType, nullptr);
}

mv::Attribute::~Attribute()
{
    
}

mv::AttrType mv::Attribute::getType() const
{
    return attrType_;
}

mv::json::Value mv::Attribute::getContentJson() const
{

    switch (attrType_)
    {

        case AttrType::UnsignedVecType:
            return Jsonable::toJsonValue(getContent<mv::dynamic_vector<unsigned>>());

        case AttrType::ByteType:
            return Jsonable::toJsonValue(getContent<byte_type>());

        case AttrType::UnsignedType:
            return Jsonable::toJsonValue(getContent<unsigned_type>());

        case AttrType::IntegerType:
            return Jsonable::toJsonValue(getContent<int_type>());

        case AttrType::FloatType:
            return Jsonable::toJsonValue(getContent<float_type>());

        case AttrType::DTypeType:
            return Jsonable::toJsonValue(getContent<DType>());

        case AttrType::OrderType:
            return Jsonable::toJsonValue(getContent<Order>());

        case AttrType::ShapeType:
            return Jsonable::toJsonValue(getContent<Shape>());

        case AttrType::StringType:
            return Jsonable::toJsonValue(getContent<string>());

        case AttrType::BoolType:
            return Jsonable::toJsonValue(getContent<bool>());

        case AttrType::OpTypeType:
            return Jsonable::toJsonValue(getContent<OpType>());

        case AttrType::FloatVec2DType:
            return Jsonable::toJsonValue(getContent<FloatVector2D>());

        case AttrType::FloatVec3DType:
            return Jsonable::toJsonValue(getContent<FloatVector3D>());

        case AttrType::FloatVec4DType:
            return Jsonable::toJsonValue(getContent<FloatVector4D>());

        case AttrType::IntVec2DType:
            return Jsonable::toJsonValue(getContent<IntVector2D>());

        case AttrType::IntVec3DType:
            return Jsonable::toJsonValue(getContent<IntVector3D>());

        case AttrType::IntVec4DType:
            return Jsonable::toJsonValue(getContent<IntVector4D>());

        case AttrType::UnsignedVec2DType:
            return Jsonable::toJsonValue(getContent<UnsignedVector2D>());

        case AttrType::UnsignedVec3DType:
            return Jsonable::toJsonValue(getContent<UnsignedVector3D>());

        case AttrType::UnsignedVec4DType:
            return Jsonable::toJsonValue(getContent<UnsignedVector4D>());

        case AttrType::FloatVecType:
            return Jsonable::toJsonValue(getContent<mv::dynamic_vector<float_type>>());

        case AttrType::StringVecType:
            return Jsonable::toJsonValue(getContent<mv::dynamic_vector<string>>());

        case AttrType::UnknownType:
            return Jsonable::toJsonValue(false);
    }

    return Jsonable::toJsonValue(false);
}

mv::string mv::Attribute::getContentStr() const
{

    switch (attrType_)
    {
        
        case AttrType::ByteType:
            return Printable::toString(getContent<byte_type>());

        case AttrType::UnsignedType:
            return Printable::toString(getContent<unsigned_type>());

        case AttrType::IntegerType:
            return Printable::toString(getContent<int_type>());

        case AttrType::FloatType:
            return Printable::toString(getContent<float_type>());

        case AttrType::DTypeType:
            return Printable::toString(getContent<DType>());

        case AttrType::OrderType:
            return Printable::toString(getContent<Order>());

        case AttrType::ShapeType:
            return Printable::toString(getContent<Shape>());

        case AttrType::StringType:
            return getContent<string>();

        case AttrType::BoolType:
            return Printable::toString(getContent<bool>());

        case AttrType::OpTypeType:
            return Printable::toString(getContent<OpType>());

        case AttrType::FloatVec2DType:
            return Printable::toString(getContent<FloatVector2D>());

        case AttrType::FloatVec3DType:
            return Printable::toString(getContent<FloatVector3D>());

        case AttrType::FloatVec4DType:
            return Printable::toString(getContent<FloatVector4D>());

        case AttrType::IntVec2DType:
            return Printable::toString(getContent<IntVector2D>());

        case AttrType::IntVec3DType:
            return Printable::toString(getContent<IntVector3D>());

        case AttrType::IntVec4DType:
            return Printable::toString(getContent<IntVector4D>());

        case AttrType::UnsignedVec2DType:
            return Printable::toString(getContent<UnsignedVector2D>());

        case AttrType::UnsignedVec3DType:
            return Printable::toString(getContent<UnsignedVector3D>());

        case AttrType::UnsignedVec4DType:
            return Printable::toString(getContent<UnsignedVector4D>());

        case AttrType::FloatVecType:
            return Printable::toString(getContent<mv::dynamic_vector<float_type>>());

        case AttrType::StringVecType:
            return Printable::toString(getContent<mv::dynamic_vector<std::string>>());

        default:
            return "unknown";

    }

}

mv::string mv::Attribute::toString() const
{
    return "(" + Printable::toString(attrType_) + "): " + getContentStr();
}

mv::json::Value mv::Attribute::toJsonValue() const
{
    mv::json::Object obj;
    obj["attrType"] = mv::Jsonable::toJsonValue(attrType_);
    obj["content"] = mv::json::Value(getContentJson());
    return mv::json::Value(obj);
}
