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

mv::json::Array::Array(const Array& other) :
elements_(other.elements_)
{


}

void mv::json::Array::append(const Value& value)
{
    elements_.push_back(value);
}

void mv::json::Array::erase(unsigned idx)
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

        output += it->stringify();
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

mv::json::Value& mv::json::Array::operator[](unsigned idx)
{

    if (idx >= size())
    {
        throw IndexError("Index out of range");
    }

    return elements_[idx];

}

mv::json::Array& mv::json::Array::operator=(const Array& other)
{
    elements_ = other.elements_;
    return *this;
}