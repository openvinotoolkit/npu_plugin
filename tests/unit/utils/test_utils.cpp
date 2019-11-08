#include "test_utils.hpp"

#include <cassert>
#include <string>

std::string testToString(const mv::Shape& shape)
{
    std::string str = std::to_string(shape[0]);
    for (unsigned i=1; i < shape.ndims(); i++)
    {
        str += "x" + std::to_string(shape[i]);
    }
    return str;
}

std::string testToString(const mv::Target& target)
{
    assert(target == mv::Target::ma2480 || target == mv::Target::ma2490);
    std::string targetName = target == mv::Target::ma2480 ? "ma2480" : "ma2490";
    return targetName;
}
