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
