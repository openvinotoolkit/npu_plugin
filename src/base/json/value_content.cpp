#include "include/mcm/base/json/value_content.hpp"

mv::json::detail::ValueContent::~ValueContent()
{

}

mv::json::detail::ValueContent::operator float&()
{
    throw ValueError("Unable to obtain a float content from a JSON value");
}

mv::json::detail::ValueContent::operator long long&()
{
    throw ValueError("Unable to obtain an int content from a JSON value");
}

mv::json::detail::ValueContent::operator std::string&()
{
    throw ValueError("Unable to obtain a string content from a JSON value");
}

mv::json::detail::ValueContent::operator bool&()
{
    throw ValueError("Unable to obtain a bool content from a JSON value");
}

mv::json::detail::ValueContent::operator Object&()
{
    throw ValueError("Unable to obtain a JSON object content from a JSON value");
}

mv::json::detail::ValueContent::operator Array&()
{
    throw ValueError("Unable to obtain a JSON array content from a JSON value");
}
