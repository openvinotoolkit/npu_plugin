#include "include/mcm/base/printable.hpp"

mv::Printable::~Printable()
{
    
}

void mv::Printable::replaceSub(string &input, const string &oldSub, const string &newSub)
{
    string::size_type pos = 0u;
    while((pos = input.find(oldSub, pos)) != string::npos)
    {
        input.replace(pos, oldSub.length(), newSub);
        pos += newSub.length();
    }
}

mv::string mv::Printable::toString(const Printable &obj)
{
    return obj.toString();
}

mv::string mv::Printable::toString(int_type value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(float_type value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(unsigned_type value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(long long value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(std::size_t value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(byte_type value)
{
    return toString((unsigned_type)value);
}

mv::string mv::Printable::toString(dim_type value)
{
    return toString((unsigned_type)value);
}

mv::string mv::Printable::toString(bool value)
{
    return std::to_string(value);
}

mv::string mv::Printable::toString(DType value)
{
    return mv::dtypeStrings.at(value);
}

mv::string mv::Printable::toString(Order value)
{
    return mv::orderStrings.at(value);
}

mv::string mv::Printable::toString(const mv::dynamic_vector<float> &value)
{
    return "(" + toString((unsigned_type)value.size()) + ")";
}

mv::string mv::Printable::toString(const mv::dynamic_vector<std::string> &value)
{

    std::string output = "(";

    if (value.size() > 0)
    {
        output += value[0];

        for (std::size_t i = 1; i < value.size(); ++i)
        {
            output += ", " + value[i];
        }

    }

    return output + ")";

}

mv::string mv::Printable::toString(AttrType value)
{
    return mv::attrTypeStrings.at(value);
}

mv::string mv::Printable::toString(OpType value)
{
    return mv::opsStrings.at(value);
}
