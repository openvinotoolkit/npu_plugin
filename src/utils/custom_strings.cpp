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

std::string mv::createSparsityMapName(const std::string& opName)
{
    return demanglePOCName(opName) + "_sparse_dw";
}

std::string mv::createWeightTableName(const std::string& opName)
{
    return demanglePOCName(opName) + "_weights_table";
}

std::string mv::createDPUTaskName(const std::string& opName)
{
    return demanglePOCName(opName);
}

std::string mv::createDeallocationName(const std::string& opName)
{
    return demanglePOCName(opName) + "_DEALLOC";
}

std::string mv::createDMATaskCMX2DDRName(const std::string& opName)
{
    return demanglePOCName(opName) + "_CMX2DDR";
}

std::string mv::createDMATaskDDR2CMXName(const std::string& opName)
{
    return demanglePOCName(opName) + "_DDR2CMX";
}

std::string mv::createAlignConstantName(const std::string& opName)
{
    return "AlignContainer_" + demanglePOCName(opName);
}

