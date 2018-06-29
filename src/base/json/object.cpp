#include "include/mcm/base/json/object.hpp"

mv::json::Object::Object()
{
    
}

bool mv::json::Object::emplace(const std::string& key, float value)
{
    std::unique_ptr<NumberFloat> ptr(new NumberFloat(value));
    members_.emplace(key, std::move(ptr));
    return true;
}

bool mv::json::Object::emplace(const std::string& key, int value)
{
    std::unique_ptr<NumberInteger> ptr(new NumberInteger(value));
    members_.emplace(key, std::move(ptr));
    return true;
}

bool mv::json::Object::emplace(const std::string& key, const std::string& value)
{
    return true;
}

bool mv::json::Object::emplace(const std::string& key, bool value)
{
    return true;
}

bool mv::json::Object::emplace(const std::string& key)
{
    std::unique_ptr<Null> ptr(new Null());
    members_.emplace(key, std::move(ptr));
    return true;
}

unsigned mv::json::Object::size() const
{
    return members_.size();
}