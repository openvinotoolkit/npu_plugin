#include "include/mcm/base/json/object.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/null.hpp"
#include "include/mcm/base/json/array.hpp"

mv::json::Object::Object() :
Value()
{
    
}

mv::json::Object::Object(const Object& other) :
Value(other)
{

    deepCopyMembers_(other.members_);

}

mv::json::Object::Object(Object& owner, const std::string& key) :
Value(owner, key, JSONType::Object)
{
    
}

void mv::json::Object::deepCopyMembers_(const std::map<std::string, std::unique_ptr<Value>>& input)
{
    
    for (const auto& entry : input)
    {

        std::string key = entry.first;
        switch (entry.second->valueType())
        {

            case JSONType::Array:
                break;

            case JSONType::Bool:
            {
                auto ptr = std::unique_ptr<Bool>(new Bool(*this, key, entry.second->get<bool>()));
                members_.emplace(key, std::move(ptr));
                break;
            }

            case JSONType::Null:
            {
                auto ptr = std::unique_ptr<Null>(new Null(*this, key));
                members_.emplace(key, std::move(ptr));
                break;
            }

            case JSONType::NumberFloat:
            {
                auto ptr = std::unique_ptr<NumberFloat>(new NumberFloat(*this, key, entry.second->get<float>()));
                members_.emplace(key, std::move(ptr));
                break;
            }

            case JSONType::NumberInteger:
            {
                auto ptr = std::unique_ptr<NumberInteger>(new NumberInteger(*this, key, entry.second->get<int>()));
                members_.emplace(key, std::move(ptr));
                break;
            }

            case JSONType::Object:
            {
                auto ptr = std::unique_ptr<Object>(new Object(entry.second->get<Object>()));
                members_.emplace(key, std::move(ptr));
                break;
            }

            case JSONType::String:
            {
                auto ptr = std::unique_ptr<String>(new String(*this, key, entry.second->get<std::string>()));
                members_.emplace(key, std::move(ptr));
                break;
            }

            default:
                throw ValueError("Unable to copy value of an unknown type");

        }

    }

}

bool mv::json::Object::emplace(const std::string& key, float value)
{
    std::unique_ptr<NumberFloat> ptr(new NumberFloat(*this, key, value));
    members_.emplace(key, std::move(ptr));
    return true;
}

bool mv::json::Object::emplace(const std::string& key, int value)
{
    std::unique_ptr<NumberInteger> ptr(new NumberInteger(*this, key, value));
    members_.emplace(key, std::move(ptr));
    return true;
}

bool mv::json::Object::emplace(const std::string& key, const std::string& value)
{
    std::unique_ptr<String> ptr(new String(*this, key, value));
    members_.emplace(key, std::move(ptr));
    return true;
}

bool mv::json::Object::emplace(const std::string& key, bool value)
{
    std::unique_ptr<Bool> ptr(new Bool(*this, key, value));
    members_.emplace(key, std::move(ptr));
    return true;
}

bool mv::json::Object::emplace(const std::string& key, const Object& value)
{
    std::unique_ptr<Object> ptr(new Object(value));
    members_.emplace(key, std::move(ptr));
    return true;
}

bool mv::json::Object::emplace(const std::string& key)
{
    std::unique_ptr<Null> ptr(new Null(*this, key));
    members_.emplace(key, std::move(ptr));
    return true;
}

void mv::json::Object::erase(const std::string& key)
{
    members_.erase(key);
}

unsigned mv::json::Object::size() const
{
    return members_.size();
}

std::string mv::json::Object::stringify() const
{

    std::string output = "{";

    auto it = members_.begin();

    if (it != members_.end())
    {

        auto str = [&it](){ return "\"" + it->first + "\":" + it->second->stringify(); };
        output += str();
        ++it;

        for (; it != members_.end(); ++it)
            output += "," + str();

    }

    output += "}";
    return output;
    
}

mv::json::Value& mv::json::Object::operator[](const std::string& key)
{

    if (members_.find(key) == members_.end())
    {
        emplace(key);
    }

    return *members_[key].get();

}

mv::json::Object& mv::json::Object::operator=(const Object& other)
{
    deepCopyMembers_(other.members_);
    return *this;
}