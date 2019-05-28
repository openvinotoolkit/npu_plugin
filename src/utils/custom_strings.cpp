#include "mcm/utils/custom_strings.hpp"

std::string mv::deleteTillEndIfPatternFound(const std::string& input, const std::string& pattern)
{
    std::string toReturn(input);
    auto patternFound = toReturn.rfind(pattern);
    if(patternFound != std::string::npos)
        toReturn.erase(patternFound);
    return toReturn;
}

std::string mv::demanglePOCName(const std::string &mangledName)
{
    return deleteTillEndIfPatternFound(mangledName, "#");
}
