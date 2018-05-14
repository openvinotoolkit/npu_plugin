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

mv::string mv::Attribute::toString() const
{
    string result = "(" + Printable::toString(attrType_) + "): ";

    switch (attrType_)
    {
        
        case AttrType::ByteType:
            result += Printable::toString(getContent<byte_type>());
            break;

        case AttrType::UnsingedType:
            result += Printable::toString(getContent<unsigned_type>());
            break;

        case AttrType::IntegerType:
            result += Printable::toString(getContent<int_type>());
            break;

        case AttrType::FloatType:
            result += Printable::toString(getContent<float_type>());
            break;

        case AttrType::TensorType:
            result += Printable::toString(getContent<ConstantTensor>());
            break;
        
        case AttrType::DTypeType:
            result += Printable::toString(getContent<DType>());
            break;

        case AttrType::OrderType:
            result += Printable::toString(getContent<Order>());
            break;

        case AttrType::ShapeType:
            result += Printable::toString(getContent<Shape>());
            break;

        case AttrType::StringType:
            result += getContent<string>();
            break;

        default:
            result += ": unknown";
            break;

    }

    return result;

}