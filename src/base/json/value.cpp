#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/object.hpp"

const std::map<mv::json::JSONType, std::string> mv::json::Value::typeString_ =
{
    {mv::json::JSONType::Array, "array"},
    {mv::json::JSONType::Bool, "bool"},
    {mv::json::JSONType::Null, "null"},
    {mv::json::JSONType::NumberFloat, "number (float)"},
    {mv::json::JSONType::NumberInteger, "number (integer)"},
    {mv::json::JSONType::Object, "object"},
    {mv::json::JSONType::String, "string"},
    {mv::json::JSONType::Unknown, "unknown"}

};

mv::json::Value::Value(Object& owner, const std::string& key, JSONType valueType) :
valueType_(valueType),
owner_(&owner),
key_(key)
{

}

mv::json::Value::Value() :
valueType_(JSONType::Object),
owner_(nullptr),
key_("")
{

}

mv::json::Value::Value(const Value& other) :
valueType_(other.valueType_),
owner_(other.owner_),
key_(other.key_)
{

}

mv::json::Value::~Value()
{
    
}

mv::json::Value::operator float&()
{
    throw ValueError("Unable to obtain a float content from a JSON value of type " + typeString_.at(valueType_));
}

mv::json::Value::operator int&()
{
    throw ValueError("Unable to obtain an int content from a JSON value of type " + typeString_.at(valueType_));
}

mv::json::Value::operator std::string&()
{
    throw ValueError("Unable to obtain a string content from a JSON value of type " + typeString_.at(valueType_));
}

mv::json::Value::operator bool&()
{
    throw ValueError("Unable to obtain a bool content from a JSON value of type " + typeString_.at(valueType_));
}

mv::json::Value::operator Object&()
{
    throw ValueError("Unable to obtain a JSON object content from a JSON value of type " + typeString_.at(valueType_));
}

mv::json::Value& mv::json::Value::operator=(float value)
{
    
    if (owner_ == nullptr)
    {
        throw ValueError("Unable to assign a value with undefined owner object");
    }

    // Postpone the deletion of this object until the end of the current scope
    auto ptr = std::move(owner_->members_[key_]);
    owner_->erase(key_);
    owner_->emplace(key_, value);
    return (*owner_)[key_];

}

mv::json::Value& mv::json::Value::operator=(int value)
{

    if (owner_ == nullptr)
    {
        throw ValueError("Unable to assign a value with undefined owner object");
    }

    // Postpone the deletion of this object until the end of the current scope
    auto ptr = std::move(owner_->members_[key_]);
    owner_->erase(key_);
    owner_->emplace(key_, value);
    return (*owner_)[key_];

}

mv::json::Value& mv::json::Value::operator=(const std::string& value)
{

    if (owner_ == nullptr)
    {
        throw ValueError("Unable to assign a value with undefined owner object");
    }

    // Postpone the deletion of this object until the end of the current scope
    auto ptr = std::move(owner_->members_[key_]);
    owner_->erase(key_);
    owner_->emplace(key_, value);
    return (*owner_)[key_];

}

mv::json::Value& mv::json::Value::operator=(bool value)
{

    if (owner_ == nullptr)
    {
        throw ValueError("Unable to assign a value with undefined owner object");
    }

    // Postpone the deletion of this object until the end of the current scope
    auto ptr = std::move(owner_->members_[key_]);
    owner_->erase(key_);
    owner_->emplace(key_, value);
    return (*owner_)[key_];

}

mv::json::Value& mv::json::Value::operator=(const Object& value)
{

    if (owner_ == nullptr)
    {
        throw ValueError("Unable to assign a value with undefined owner object");
    }

    // Postpone the deletion of this object until the end of the current scope
    auto ptr = std::move(owner_->members_[key_]);
    owner_->erase(key_);
    owner_->emplace(key_, value);
    return (*owner_)[key_];

}

mv::json::Value& mv::json::Value::operator[](const std::string& key)
{

    if (valueType_ == JSONType::Object)
    {
        auto objPtr = static_cast<Object*>(this);
        return (*objPtr)[key];
    }

    Value& objValue = operator=(Object(*owner_, key));
    return objValue[key];

}

mv::json::JSONType mv::json::Value::valueType() const
{
    return valueType_;
}

mv::json::Value& mv::json::Value::operator=(const Value& other)
{
    valueType_ = other.valueType_;
    owner_ = other.owner_;
    key_ = other.key_;
    return *this;
}