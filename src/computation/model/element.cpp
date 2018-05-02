#include "include/fathom/computation/model/element.hpp"

mv::allocator mv::ComputationElement::allocator_;

mv::byte_type mv::ComputationElement::Attribute::nextTypeId()
{
    static byte_type id(0);
    assert(id < max_byte && "Out of attribute types ID");
    return id++;
}

mv::ComputationElement::ComputationElement(const Logger &logger, const std::string &name) : 
logger_(logger),
name_(name),
attributes_(allocator_)
{

}

mv::ComputationElement::~ComputationElement()
{
    
}

const std::string &mv::ComputationElement::getName() const
{
    return name_;
}