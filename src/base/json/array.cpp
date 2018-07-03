#include "include/mcm/base/json/array.hpp"
#include "include/mcm/base/json/object.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/null.hpp"

mv::json::Array::Array() :
Value()
{

}

mv::json::Array::Array(const Array& other) :
Value(other)
{

}

mv::json::Array::Array(Object& owner, const std::string& key) :
Value(owner, key, JSONType::Array)
{

}

void mv::json::Array::deepCopyMembers_(const std::vector<std::unique_ptr<Value>>& input)
{

    for (const auto& entry : input)
    {

        switch (entry->valueType())
        {

            /*case JSONType::Array:
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
                throw ValueError("Unable to copy value of an unknown type");*/

        }

    }

}

void mv::json::Array::erase(unsigned idx)
{
    elements_.erase(elements_.begin() + idx);
}

unsigned mv::json::Array::size() const
{
    return elements_.size();
}

mv::json::Value& mv::json::Array::operator[](unsigned idx)
{

    if (idx > size())
        throw IndexError("Index out of range");

    return *elements_[idx].get();
    
}

std::string mv::json::Array::stringify() const
{

}

mv::json::Array& mv::json::Array::operator=(const Array& other)
{
    
}