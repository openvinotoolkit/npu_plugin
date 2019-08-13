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
    std::string createDMATaskNNCMX2DDRName(const std::string& opName);
    std::string createDMATaskDDR2NNCMXName(const std::string& opName);
    std::string createDMATaskUPACMX2NNCMXName(const std::string& opName);
    std::string createDMATaskUPACMX2DDRName(const std::string& opName);
    std::string createDMATaskDDR2UPACMXName(const std::string& opName);
    std::string createDMATaskNNCMX2UPACMXName(const std::string& opName);
    std::string createAlignConstantName(const std::string& opName);
    std::string createAlignWeightSetConstantName(const std::string& opName);
    std::string createBarrierName(const std::string&, unsigned barrierID);
    std::string createBiasName(const std::string& opName);
}

#endif
