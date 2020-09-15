#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/object.hpp"
#include "include/mcm/base/json/array.hpp"
#include "include/mcm/base/json/null.hpp"

const std::map<mv::json::JSONType, std::string> mv::json::Value::typeString_ =
{
    {JSONType::Array, "json::Array"},
    {JSONType::Bool, "json::Bool"},
    {JSONType::Null, "json::Null"},
    {JSONType::NumberFloat, "json::NumberFloat"},
    {JSONType::NumberInteger, "json::NumberInteger"},
    {JSONType::Object, "json::Object"},
    {JSONType::String, "json::String"}

};

mv::json::Value::Value() :
valueType_(JSONType::Null),
content_(std::unique_ptr<Null>(new Null()))
{

}

mv::json::Value::Value(double value) :
valueType_(JSONType::NumberFloat),
content_(std::unique_ptr<NumberFloat>(new NumberFloat(value)))
{

}

mv::json::Value::Value(long long value) :
valueType_(JSONType::NumberInteger),
content_(std::unique_ptr<NumberInteger>(new NumberInteger(value)))
{

}

mv::json::Value::Value(const char * value) :
valueType_(JSONType::String),
content_(std::unique_ptr<String>(new String(value)))
{

}

mv::json::Value::Value(const std::string& value) :
valueType_(JSONType::String),
content_(std::unique_ptr<String>(new String(value)))
{

}

mv::json::Value::Value(bool value) :
valueType_(JSONType::Bool),
content_(std::unique_ptr<Bool>(new Bool(value)))
{

}

mv::json::Value::Value(const Object& value) :
valueType_(JSONType::Object),
content_(std::unique_ptr<Object>(new Object(value)))
{

}

mv::json::Value::Value(const Array& value) :
valueType_(JSONType::Array),
content_(std::unique_ptr<Array>(new Array(value)))
{

}

mv::json::Value::Value(const Value& other)
{
    operator=(other);
}

mv::json::Value::~Value()
{
    
}

mv::json::Value& mv::json::Value::operator=(double value)
{

    content_.reset();
    content_ = std::unique_ptr<NumberFloat>(new NumberFloat(value));
    valueType_ = JSONType::NumberFloat;
    return *this;

}

mv::json::Value& mv::json::Value::operator=(long long value)
{

    content_.reset();
    content_ = std::unique_ptr<NumberInteger>(new NumberInteger(value));
    valueType_ = JSONType::NumberInteger;
    return *this;

}

mv::json::Value& mv::json::Value::operator=(const std::string& value)
{

    content_.reset();
    content_ = std::unique_ptr<String>(new String(value));
    valueType_ = JSONType::String;
    return *this;

}

mv::json::Value& mv::json::Value::operator=(bool value)
{

    content_.reset();
    content_ = std::unique_ptr<Bool>(new Bool(value));
    valueType_ = JSONType::Bool;
    return *this;

}

mv::json::Value& mv::json::Value::operator=(const Object& value)
{

    content_.reset();
    content_ = std::unique_ptr<Object>(new Object(value));
    valueType_ = JSONType::Object;
    return *this;

}

mv::json::Value& mv::json::Value::operator=(const Array& value)
{

    content_.reset();
    content_ = std::unique_ptr<Array>(new Array(value));
    valueType_ = JSONType::Array;
    return *this;

}

bool mv::json::Value::operator==(const Value& other) const
{

    if (valueType() != other.valueType())
        return false;

    switch (valueType())
    {

        case JSONType::Array:
            if (*static_cast<json::Array*>(content_.get()) != *static_cast<json::Array*>(other.content_.get()))
                return false;
            break;
        
        case JSONType::Bool:
            if (*static_cast<json::Bool*>(content_.get()) != *static_cast<json::Bool*>(other.content_.get()))
                return false;
            break;

        case JSONType::Null:
            // Nulls are always equal
            break;

        case JSONType::NumberFloat:
            if (*static_cast<json::NumberFloat*>(content_.get()) != *static_cast<json::NumberFloat*>(other.content_.get()))
                return false;
            break;

        case JSONType::NumberInteger:
            if (*static_cast<json::NumberInteger*>(content_.get()) != *static_cast<json::NumberInteger*>(other.content_.get()))
                return false;
            break;

        case JSONType::Object:
            if (*static_cast<json::Object*>(content_.get()) != *static_cast<json::Object*>(other.content_.get()))
                return false;
            break;

        case JSONType::String:
            if (*static_cast<json::String*>(content_.get()) != *static_cast<json::String*>(other.content_.get()))
                return false;
            break;

    }

    return true;

}

bool mv::json::Value::operator!=(const Value& other) const
{
    return !operator==(other);
}

mv::json::Value& mv::json::Value::operator[](const std::string& key)
{

    if (valueType_ == JSONType::Object)
    {
        auto objPtr = static_cast<Object*>(content_.get());
        return (*objPtr)[key];
    }

    Value& objValue = operator=(Object());
    return objValue[key];

}

const mv::json::Value& mv::json::Value::operator[](const std::string& key) const
{

    if (valueType_ != JSONType::Object)
    {
         throw ValueError(*this, "Attempt of accessing the content of value " + typeString_.at(valueType_) + " as to JSON object");
    }
    auto objPtr = static_cast<Object*>(content_.get());
    return (*objPtr)[key];

}

mv::json::Value& mv::json::Value::operator[](std::size_t idx)
{

    if (valueType_ != JSONType::Array)
        throw ValueError(*this, "Attempt of accessing the content of value " + typeString_.at(valueType_) + " as to JSON array");

    auto arrPtr = static_cast<Array*>(content_.get());
    return (*arrPtr)[idx];
    
}

const mv::json::Value& mv::json::Value::operator[](std::size_t idx) const
{

    if (valueType_ != JSONType::Array)
        throw ValueError(*this, "Attempt of accessing the content of value " + typeString_.at(valueType_) + " as to JSON array");

    auto arrPtr = static_cast<Array*>(content_.get());
    return (*arrPtr)[idx];
    
}


mv::json::Value& mv::json::Value::last()
{
    if (valueType_ != JSONType::Array)
        throw ValueError(*this, "Attempt of accessing the content of value " + typeString_.at(valueType_) + " as to JSON array");

    auto arrPtr = static_cast<Array*>(content_.get());
    return arrPtr->last();
}

bool mv::json::Value::hasKey(const std::string& key) const
{
    if (valueType_ != JSONType::Object)
        throw ValueError(*this, "Attempt of checking key for a value of type " + typeString_.at(valueType_));
    
    auto objPtr = static_cast<Object*>(content_.get());
    return objPtr->hasKey(key);
}

std::vector<std::string> mv::json::Value::getKeys() const
{
    if (valueType_ != JSONType::Object)
        throw ValueError(*this, "Attempt of obtaining the keys list from a value of type " + typeString_.at(valueType_));
    
    auto objPtr = static_cast<Object*>(content_.get());
    return objPtr->getKeys();
}

void mv::json::Value::emplace(const std::pair<std::string, Value>& member)
{
    if (valueType_ != JSONType::Object)
        throw ValueError(*this, "Attempt of appending a memeber content to the value of type " + typeString_.at(valueType_));

    auto objPtr = static_cast<Object*>(content_.get());
    (*objPtr)[member.first] = member.second;
}

void mv::json::Value::append(const Value& element)
{
    if (valueType_ != JSONType::Array)
        throw ValueError(*this, "Attempt of appending an element content to the value of type " + typeString_.at(valueType_));

    auto arrPtr = static_cast<Array*>(content_.get());
    arrPtr->append(element);
}

mv::json::JSONType mv::json::Value::valueType() const
{
    return valueType_;
}

std::string mv::json::Value::typeName(JSONType typeID)
{
    return typeString_.at(typeID);
}

std::size_t mv::json::Value::size() const
{
    if (valueType_ == JSONType::Array)
    {
        auto arrPtr = static_cast<Array*>(content_.get());
        return arrPtr->size();
    }

    if (valueType_ == JSONType::Object)
    {
        auto objPtr = static_cast<Object*>(content_.get());
        return objPtr->size();
    }

    return 0;
}

std::string mv::json::Value::stringify() const
{
    return content_->stringify();
}

std::string mv::json::Value::stringifyPretty() const
{
    return content_->stringifyPretty();
}

mv::json::Value& mv::json::Value::operator=(const Value& other)
{
    valueType_ = other.valueType_;
    content_.reset();
    switch (valueType_)
    {

        case mv::json::JSONType::Array:
            content_ = std::unique_ptr<Array>(new Array(*(Array*)other.content_.get()));
            break;

        case mv::json::JSONType::Bool:
            content_ = std::unique_ptr<Bool>(new Bool((bool)*other.content_.get()));
            break;

        case mv::json::JSONType::Null:
            content_ = std::unique_ptr<Null>(new Null());
            break;

        case mv::json::JSONType::NumberFloat:
            content_ = std::unique_ptr<NumberFloat>(new NumberFloat((double)*other.content_.get()));
            break;

        case mv::json::JSONType::NumberInteger:
            content_ = std::unique_ptr<NumberInteger>(new NumberInteger((long long)*other.content_.get()));
            break;
        
        case mv::json::JSONType::Object:
            content_ = std::unique_ptr<Object>(new Object(*(Object*)other.content_.get()));
            break;

        case mv::json::JSONType::String:
            content_ = std::unique_ptr<String>(new String((std::string)*other.content_.get()));
            break;

    }
    
    return *this;
}

std::string mv::json::Value::getLogID() const
{
    return "json::Value";
}