#include "include/mcm/base/printable.hpp"

mv::Printable::~Printable()
{
    
}

void mv::Printable::replaceSub(std::string &input, const std::string &oldSub, const std::string &newSub)
{
    std::string::size_type pos = 0u;
    while((pos = input.find(oldSub, pos)) != std::string::npos)
    {
        input.replace(pos, oldSub.length(), newSub);
        pos += newSub.length();
    }
}

std::string mv::Printable::toString(const Printable &obj)
{
    return obj.toString();
}

std::string mv::Printable::toString(int value)
{
    return std::to_string(value);
}

std::string mv::Printable::toString(double value)
{
    return std::to_string(value);
}

std::string mv::Printable::toString(unsigned value)
{
    return std::to_string(value);
}

std::string mv::Printable::toString(unsigned long long value)
{
    return std::to_string(value);
}

std::string mv::Printable::toString(std::size_t value)
{
    return std::to_string(value);
}

std::string mv::Printable::toString(bool value)
{
    return std::to_string(value);
}

std::string mv::Printable::toString(DType value)
{
    return mv::dtypeStrings.at(value);
}

std::string mv::Printable::toString(Order value)
{
    return mv::orderStrings.at(value);
}

std::string mv::Printable::toString(const std::vector<double> &value)
{
    return "(" + toString(value.size()) + ")";
}

std::string mv::Printable::toString(const std::vector<std::string> &value)
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

std::string mv::Printable::toString(AttrType value)
{
    return mv::attrTypeStrings.at(value);
}

std::string mv::Printable::toString(OpType value)
{
    return mv::opsStrings.at(value);
}
