#include "include/fathom/computation/model/attribute.hpp"

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

        case AttrType::TensorType:
            return Printable::toString(getContent<ConstantTensor>());

        case AttrType::DTypeType:
            return Printable::toString(getContent<DType>());

        case AttrType::OrderType:
            return Printable::toString(getContent<Order>());

        case AttrType::ShapeType:
            return Printable::toString(getContent<Shape>());

        case AttrType::StringType:
            return getContent<string>();

        default:
            return "unknown";

    }

}

mv::string mv::Attribute::toString() const
{
    
    return "(" + Printable::toString(attrType_) + "): " + getContentStr();

}