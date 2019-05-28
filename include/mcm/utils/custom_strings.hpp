#ifndef CUSTOM_STRINGS_HPP
#define CUSTOM_STRINGS_HPP

#include <string>

namespace mv
{
    std::string demanglePOCName(const std::string& mangledName);
    std::string deleteTillEndIfPatternFound(const std::string& input, const std::string& pattern);

}

#endif
