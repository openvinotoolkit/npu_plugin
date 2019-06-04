#ifndef CUSTOM_STRINGS_HPP
#define CUSTOM_STRINGS_HPP

#include <string>

namespace mv
{
    std::string demanglePOCName(const std::string& mangledName);
    std::string deleteTillEndIfPatternFound(const std::string& input, const std::string& pattern);
    std::string createFakeSparsityMapName(const std::string& opName);
    std::string createSparsityMapName(const std::string& tensorName);
    std::string createStorageElementName(const std::string& tensorName);
    std::string createWeightTableName(const std::string& opName);
    std::string createDPUTaskName(const std::string& opName);
    std::string createDeallocationName(const std::string& opName);
    std::string createDMATaskCMX2DDRName(const std::string& opName);
    std::string createDMATaskDDR2CMXName(const std::string& opName);
    std::string createAlignConstantName(const std::string& opName);
    std::string createBarrierName(const std::string& opName, unsigned barrierID);

}

#endif
