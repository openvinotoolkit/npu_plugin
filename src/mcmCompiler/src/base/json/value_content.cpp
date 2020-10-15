#include "include/mcm/base/json/value_content.hpp"

mv::json::detail::ValueContent::~ValueContent()
{

}

mv::json::detail::ValueContent::operator double&()
{
    throw ValueError(*this, "Unable to obtain a double content");
}

mv::json::detail::ValueContent::operator long long&()
{
    throw ValueError(*this, "Unable to obtain an long long content");
}

mv::json::detail::ValueContent::operator std::string&()
{
    throw ValueError(*this, "Unable to obtain a std::string content");
}

mv::json::detail::ValueContent::operator bool&()
{
    throw ValueError(*this, "Unable to obtain a bool content");
}

mv::json::detail::ValueContent::operator mv::json::Object&()
{
    throw ValueError(*this, "Unable to obtain a json::Object content");
}

mv::json::detail::ValueContent::operator mv::json::Array&()
{
    throw ValueError(*this, "Unable to obtain a json::Array content");
}
