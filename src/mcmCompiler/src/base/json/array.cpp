#include "include/mcm/base/json/array.hpp"
#include "include/mcm/base/json/object.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/null.hpp"

mv::json::Array::Array()
{
    
}

mv::json::Array::Array(std::initializer_list<Value> l) :
elements_(l)
{

}

mv::json::Array::Array(std::size_t s) :
elements_(s)
{

}

mv::json::Array::Array(const Array& other) :
elements_(other.elements_)
{


}

void mv::json::Array::append(const Value& value)
{
    elements_.push_back(value);
}

void mv::json::Array::erase(std::size_t idx)
{
    elements_.erase(elements_.begin() + idx);
}

unsigned mv::json::Array::size() const
{
    return elements_.size();
}

void mv::json::Array::clear()
{
    elements_.clear();
}

std::string mv::json::Array::stringify() const
{

    std::string output = "[";

    auto it = elements_.begin();

    if (it != elements_.end())
    {

        output += it->stringify();
        ++it;

        for (; it != elements_.end(); ++it)
            output += "," + it->stringify();

    }

    output += "]";
    return output;
    
}

std::string mv::json::Array::stringifyPretty() const
{

    std::string output = "[\n\t";

    auto it = elements_.begin();

    if (it != elements_.end())
    {
        //First element of array needs to be treated differently (unfortunately)
        std::string a;
        if (it->valueType() == JSONType::Array)
            a += it->get<Array>().stringifyPretty();
        else if (it->valueType() == JSONType::Object)
            a += it->get<Object>().stringifyPretty();
        else
            a += it->stringify();

        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (a[i] == '\n')
                a.insert(i + 1, "\t");
        }

        output += a;
        ++it;

        for (; it != elements_.end(); ++it)
        {
            std::string e;
            if (it->valueType() == JSONType::Array)
                e += ",\n" + it->get<Array>().stringifyPretty();
            else if (it->valueType() == JSONType::Object)
                e += ",\n" + it->get<Object>().stringifyPretty();
            else
                e += ",\n" + it->stringify();

            for (std::size_t i = 0; i < e.size(); ++i)
            {
                if (e[i] == '\n')
                    e.insert(i + 1, "\t");
            }

            output += e;

        }
    }

    output += "\n]";
    return output;

}

mv::json::Value& mv::json::Array::operator[](std::size_t idx)
{

    if (idx >= size())
        throw IndexError(*this, idx, "Out of range");

    return elements_[idx];

}

const mv::json::Value& mv::json::Array::operator[](std::size_t idx) const
{

    if (idx >= size())
        throw IndexError(*this, idx, "Out of range");

    return elements_[idx];

}

mv::json::Value& mv::json::Array::last()
{

    if (size() == 0)
    {
        throw ValueError(*this, "Cannot access the last element of an empty array");
    }

    return elements_[size() - 1];

}

mv::json::Array& mv::json::Array::operator=(const Array& other)
{
    elements_ = other.elements_;
    return *this;
}

bool mv::json::Array::operator==(const Array& other) const
{
    
    if (size() != other.size())
        return false;

    auto e1 = elements_.begin();
    for (auto e2 = other.elements_.begin(); e2 != other.elements_.end(); ++e2)
    {
        if (*e1 != *e2)
            return false;
        ++e1;
    }

    return true;

}

bool mv::json::Array::operator!=(const Array& other) const
{
    return !operator==(other);
}

std::string mv::json::Array::getLogID() const
{
    return "json::Array (size " + std::to_string(size()) + ")";
}