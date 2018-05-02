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

        result += "\n\t";

        switch (it->second.getType())
        {
            
            case ByteType:
                result += it->first + ": " + Printable::toString(it->second.getContent<byte_type>());
                break;

            case UnsingedType:
                result += it->first + ": " + Printable::toString(it->second.getContent<unsigned_type>());
                break;

            case IntegerType:
                result += it->first + ": " + Printable::toString(it->second.getContent<int_type>());
                break;

            case FloatType:
                result += it->first + ": " + Printable::toString(it->second.getContent<float_type>());
                break;

            case TensorType:
                result += it->first + ": " + it->second.getContent<ConstantModelTensor>().toString();
                break;
            
            default:
                result += it->first + ": unknown type";
                break;
        }
        
    }

    return result;

}