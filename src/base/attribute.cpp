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

mv::Attribute::Attribute(mv::json::Value& value)
{
    string attributeType = constructStringFromJson(value["attrType"]);
    mv::AttrType attr = mv::attrTypeStringsReversed.at(attributeType);
    switch (attr) {
    case mv::AttrType::ByteType:
        attrType_ = mv::AttrType::ByteType;
        content_ = allocator_.make_owner<AttributeContent<byte_type>>(constructByteTypeFromJson(value["content"]));
        break;
    case mv::AttrType::BoolType:
        attrType_ = mv::AttrType::BoolType;
        content_ = allocator_.make_owner<AttributeContent<bool>>(constructBoolTypeFromJson(value["content"]));
        break;
    case mv::AttrType::UnsignedType:
        attrType_ = mv::AttrType::UnsignedType;
        content_ = allocator_.make_owner<AttributeContent<unsigned_type>>(constructUnsignedTypeFromJson(value["content"]));
        break;
    case mv::AttrType::IntegerType:
        attrType_ = mv::AttrType::IntegerType;
        content_ = allocator_.make_owner<AttributeContent<int_type>>(constructIntTypeFromJson(value["content"]));
        break;
    case mv::AttrType::FloatType:
        attrType_ = mv::AttrType::FloatType;
        content_ = allocator_.make_owner<AttributeContent<float_type>>(constructFloatTypeFromJson(value["content"]));
        break;
    case mv::AttrType::DTypeType:
        attrType_ = mv::AttrType::DTypeType;
        content_ = allocator_.make_owner<AttributeContent<DType>>(constructDTypeFromJson(value["content"]));
        break;
    case mv::AttrType::OrderType:
        attrType_ = mv::AttrType::OrderType;
        content_ = allocator_.make_owner<AttributeContent<Order>>(constructOrderTypeFromJson(value["content"]));
        break;
    case mv::AttrType::ShapeType:
        attrType_ = mv::AttrType::ShapeType;
        content_ = allocator_.make_owner<AttributeContent<Shape>>(Shape(value["content"]));
        break;
    case mv::AttrType::StringType:
        attrType_ = mv::AttrType::StringType;
        content_ = allocator_.make_owner<AttributeContent<string>>(constructStringFromJson(value["content"]));
        break;
    case mv::AttrType::OpTypeType:
        attrType_ = mv::AttrType::OpTypeType;
        content_ = allocator_.make_owner<AttributeContent<OpType>>(constructOpTypeFromJson(value["content"]));
        break;
    case mv::AttrType::FloatVec2DType:
        attrType_ = mv::AttrType::FloatVec2DType;
        content_ = allocator_.make_owner<AttributeContent<FloatVector2D>>(constructFloatVector2DFromJson(value["content"]));
        break;
    case mv::AttrType::FloatVec3DType:
        attrType_ = mv::AttrType::FloatVec3DType;
        content_ = allocator_.make_owner<AttributeContent<FloatVector3D>>(constructFloatVector3DFromJson(value["content"]));
        break;
    case mv::AttrType::FloatVec4DType:
        attrType_ = mv::AttrType::FloatVec4DType;
        content_ = allocator_.make_owner<AttributeContent<FloatVector4D>>(constructFloatVector4DFromJson(value["content"]));
        break;
    case mv::AttrType::IntVec2DType:
        attrType_ = mv::AttrType::IntVec2DType;
        content_ = allocator_.make_owner<AttributeContent<IntVector2D>>(constructIntVector2DFromJson(value["content"]));
        break;
    case mv::AttrType::IntVec3DType:
        attrType_ = mv::AttrType::IntVec3DType;
        content_ = allocator_.make_owner<AttributeContent<IntVector3D>>(constructIntVector3DFromJson(value["content"]));
        break;
    case mv::AttrType::IntVec4DType:
        attrType_ = mv::AttrType::IntVec4DType;
        content_ = allocator_.make_owner<AttributeContent<IntVector4D>>(constructIntVector4DFromJson(value["content"]));
        break;
    case mv::AttrType::UnsignedVec2DType:
        attrType_ = mv::AttrType::UnsignedVec2DType;
        content_ = allocator_.make_owner<AttributeContent<UnsignedVector2D>>(constructUnsignedVector2DFromJson(value["content"]));
        break;
    case mv::AttrType::UnsignedVec3DType:
        attrType_ = mv::AttrType::UnsignedVec3DType;
        content_ = allocator_.make_owner<AttributeContent<UnsignedVector3D>>(constructUnsignedVector3DFromJson(value["content"]));
        break;
    case mv::AttrType::UnsignedVec4DType:
        attrType_ = mv::AttrType::UnsignedVec4DType;
        content_ = allocator_.make_owner<AttributeContent<UnsignedVector4D>>(constructUnsignedVector4DFromJson(value["content"]));
        break;
        /*
    case mv::AttrType::FloatVecType:
        attrType_ = mv::AttrType::FloatVecType;
        content_ = allocator_.make_owner<AttributeContent<mv::dynamic_vector<float_type>>(constructFloatVectorFromJson(value["content"]));
        break;
        */
    default:
        break;
    }
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

        default:
            return "unknown";

    }

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
