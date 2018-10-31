#include "include/mcm/base/attribute_entry.hpp"

mv::attr::AttributeEntry::AttributeEntry(const std::type_index& typeID) :
typeID_(typeID),
typeName_("UNNAMED")
{

}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setName(const std::string& typeName)
{
    typeName_ = typeName;
    Printable::replaceSub(typeName_, " , ", ", ");
    return *this;
}

const std::string& mv::attr::AttributeEntry::getTypeName() const
{
    return typeName_;
}

std::type_index mv::attr::AttributeEntry::getTypeID() const
{
    return typeID_;
}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setDescription(const std::string& description)
{
    description_ = description;
    return *this;
}

const std::string& mv::attr::AttributeEntry::getDescription() const
{
    return description_;
}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setCheckFunc(const std::function<bool(const Attribute&, std::string&)>& f)
{
    auto pFunc = std::make_shared<ConcreteFunc<bool, const Attribute&, std::string&> >();
    pFunc->f = f;
    checkFunc_ = pFunc;
    return *this;
}

const std::function<bool(const mv::Attribute&, std::string&)>& mv::attr::AttributeEntry::getCheckFunc()
{

    if (!checkFunc_)
        throw MasterError(*this, "Undefined check function for the argument type " + typeName_);
        
    auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<bool, const Attribute&, std::string&> >(checkFunc_);
    if (pFunc)
        return pFunc->f;

    throw AttributeError(*this, "Invalid types specified for check function for type " + typeName_);

}

bool mv::attr::AttributeEntry::hasCheckFunc() const
{
    return static_cast<bool>(checkFunc_);
}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setToJSONFunc(const std::function<mv::json::Value(const Attribute&)>& f)
{
    auto pFunc = std::make_shared<ConcreteFunc<mv::json::Value, const Attribute&> >();
    pFunc->f = f;
    toJSONFunc_ = pFunc;
    return *this;
}

const std::function<mv::json::Value(const mv::Attribute&)>& mv::attr::AttributeEntry::getToJSONFunc()
{

    if (!toJSONFunc_)
        throw MasterError(*this, "Undefined to-JSON conversion function for the argument type " + typeName_);

    auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<mv::json::Value, const Attribute&> >(toJSONFunc_);
    if (pFunc)
        return pFunc->f;
        
    throw AttributeError(*this, "Invalid types specified for to-JSON conversion function for type " + typeName_);

}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setFromJSONFunc(const std::function<Attribute(const mv::json::Value&)>& f)
{

    auto pFunc = std::make_shared<ConcreteFunc<Attribute, const mv::json::Value&> >();
    pFunc->f = f;
    fromJSONFunc_ = pFunc;

    return *this;

}

const std::function<mv::Attribute(const mv::json::Value&)>& mv::attr::AttributeEntry::getFromJSONFunc()
{

    if (!fromJSONFunc_)
        throw MasterError(*this, "Undefined from-JSON conversion function for the argument type " + typeName_);

    auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<Attribute, const mv::json::Value&> >(fromJSONFunc_);

    if (pFunc)
        return pFunc->f;

    throw AttributeError(*this, "Invalid types specified for from-JSON conversion function for type " + typeName_);

}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setToStringFunc(const std::function<std::string(const Attribute&)>& f)
{
    auto pFunc = std::make_shared<ConcreteFunc<std::string, const Attribute&> >();
    pFunc->f = f;
    toStringFunc_ = pFunc;
    return *this;
}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setShortToStringFunc(const std::function<std::string(const Attribute&)>& f)
{
    auto pFunc = std::make_shared<ConcreteFunc<std::string, const Attribute&> >();
    pFunc->f = f;
    toShortStringFunc_ = pFunc;
    return *this;
}


const std::function<std::string(const mv::Attribute&)>& mv::attr::AttributeEntry::getToStringFunc()
{

    if (!toStringFunc_)
        throw MasterError(*this, "Undefined to-string conversion function for the argument type ");

    auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<std::string, const Attribute&> >(toStringFunc_);
    if (pFunc)
        return pFunc->f;
        
    throw AttributeError(*this, "Invalid types specified for to-string conversion function for type " + typeName_);

}

const std::function<std::string(const mv::Attribute&)>& mv::attr::AttributeEntry::getShortToStringFunc()
{

    if (!hasTypeTrait("large"))
        throw AttributeError(*this, "Short-from to-string conversion is defined only for attributes with type trait large");

    if (!toShortStringFunc_)
        throw MasterError(*this, "Undefined to-string conversion (short-form) function for the argument type ");

    auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<std::string, const Attribute&> >(toShortStringFunc_);
    if (pFunc)
        return pFunc->f;
        
    throw AttributeError(*this, "Invalid types specified for to-string conversion (short-form) function for type " + typeName_);

}

mv::attr::AttributeEntry& mv::attr::AttributeEntry::setTypeTrait(const std::string& trait)
{
    if (typeTraits_.find(trait) == typeTraits_.end())
        typeTraits_.insert(trait);
    return *this;
}

bool mv::attr::AttributeEntry::hasTypeTrait(const std::string& trait)
{
    return typeTraits_.find(trait) != typeTraits_.end();
}

std::string mv::attr::AttributeEntry::getLogID() const
{
    return "AttributeRegistry entry " + typeName_;
}