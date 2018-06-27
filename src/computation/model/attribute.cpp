#include "include/mcm/computation/model/attribute.hpp"

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

mv::Attribute::~Attribute()
{
    
}

mv::AttrType mv::Attribute::getType() const
{
    return attrType_;
}

mv::string mv::Attribute::getContentStr() const
{

    switch (attrType_)
    {
        
        case AttrType::ByteType:
            return Printable::toString(getContent<byte_type>());

        case AttrType::UnsingedType:
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