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

std::string mv::createFakeSparsityMapName(const std::string& opName)
{
    return opName + "_sparse_dw";
}

std::string mv::createSparsityMapName(const std::string& tensorName)
{
    return tensorName + "_sm";
}

std::string mv::createStorageElementName(const std::string& tensorName)
{
    return tensorName + "_se";
}

std::string mv::createWeightTableName(const std::string& opName)
{
    return opName + "_weights_table";
}

std::string mv::createDPUTaskName(const std::string& opName)
{
    return opName;
}

std::string mv::createDeallocationName(const std::string& opName)
{
    return opName + "_DEALLOC";
}

std::string mv::createDMATaskNNCMX2DDRName(const std::string& opName)
{
    return opName + "_NNCMX2DDR";
}

std::string mv::createDMATaskDDR2NNCMXName(const std::string& opName)
{
    return opName + "_DDR2CMX";
}

std::string mv::createDMATaskUPACMX2NNCMXName(const std::string& opName)
{
    return opName + "_UPACMX2NNCMX";
}

std::string mv::createDMATaskUPACMX2DDRName(const std::string& opName)
{
    return opName + "_UPACMX2DDR";
}

std::string mv::createDMATaskDDR2UPACMXName(const std::string& opName)
{
    return opName + "_DDR2UPACMX";
}

std::string mv::createDMATaskNNCMX2UPACMXName(const std::string& opName)
{
    return opName + "_NNCMX2UPACMX";
}

std::string mv::createAlignConstantName(const std::string& opName)
{
    return opName + "_ALIGNED";
}

std::string mv::createAlignWeightSetConstantName(const std::string& opName)
{
    return "AlignContainer_" + opName;
}

//Barrier ID HAS TO BE USED!!! It is unique because of the static counter in the barrier class
std::string mv::createBarrierName(const std::string&, unsigned barrierID)
{
    return "Barrier_" + std::to_string(barrierID);
}

std::string mv::createBiasName(const std::string& opName)
{
    return opName + "_bias";
}


