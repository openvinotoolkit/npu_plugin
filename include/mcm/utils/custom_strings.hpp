#ifndef CUSTOM_STRINGS_HPP
#define CUSTOM_STRINGS_HPP

#include <string>

namespace mv
{
    std::string demanglePOCName(const std::string& mangledName);
    std::string deleteTillEndIfPatternFound(const std::string& input, const std::string& pattern);
    std::string createSparsityMapName(const std::string& opName);
    std::string createWeightTableName(const std::string& opName);
    std::string createDPUTaskName(const std::string& opName);
    std::string createDeallocationName(const std::string& opName);
    std::string createDMATaskCMX2DDRName(const std::string& opName);
    std::string createDMATaskDDR2CMXName(const std::string& opName);
    std::string createAlignConstantName(const std::string& opName);

}

#endif
