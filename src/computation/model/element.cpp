#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/tensor/model_constant.hpp"

mv::allocator mv::ComputationElement::allocator_;

mv::byte_type mv::ComputationElement::Attribute::nextTypeId()
{
    static byte_type id(0);
    assert(id < max_byte && "Out of attribute types ID");
    return id++;
}

mv::ComputationElement::ComputationElement(const Logger &logger, const string &name) : 
logger_(logger),
name_(name),
attributes_(allocator_)
{

}

mv::ComputationElement::~ComputationElement()
{
    
}

const mv::string &mv::ComputationElement::getName() const
{
    return name_;
}

mv::string mv::ComputationElement::toString() const
{
    string result;

    for (auto it = attributes_.begin(); it != attributes_.end(); ++it)
    {

        result += "\n\t\t";

        switch (it->second.getType())
        {
            
            case AttrType::ByteType:
                result += it->first + " (byte): " + Printable::toString(it->second.getContent<byte_type>());
                break;

            case AttrType::UnsingedType:
                result += it->first + " (unsigned): " + Printable::toString(it->second.getContent<unsigned_type>());
                break;

            case AttrType::IntegerType:
                result += it->first + " (int): " + Printable::toString(it->second.getContent<int_type>());
                break;

            case AttrType::FloatType:
                result += it->first + " (float): " + Printable::toString(it->second.getContent<float_type>());
                break;

            case AttrType::TensorType:
                result += it->first + " (const tensor): " + Printable::toString(it->second.getContent<ConstantModelTensor>());
                break;
            
            case AttrType::DTypeType:
                result += it->first + " (dType): " + Printable::toString(it->second.getContent<DType>());
                break;

            case AttrType::OrderType:
                result += it->first + " (order): " + Printable::toString(it->second.getContent<Order>());
                break;

            case AttrType::ShapeType:
                result += it->first + " (shape): " + Printable::toString(it->second.getContent<Shape>());
                break;

            default:
                result += it->first + ": unknown type";
                break;
        }
        
    }

    return result;

}