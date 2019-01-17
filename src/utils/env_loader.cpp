#include "include/mcm/utils/env_loader.hpp"
#include <iostream>
#include <fstream>

std::string mv::utils::mdkRootPath()
{

    std::string path = std::getenv("MDK_HOME");
    if (*path.rbegin() == '/')
        path.erase(path.size() - 1);
    return path;

}


std::string mv::utils::projectRootPath()
{
    
    std::string path = std::getenv("MCM_HOME");
    if (*path.rbegin() == '/')
        path.erase(path.size() - 1);
    return path;

}

bool mv::utils::fileExists(const std::string& fileName)
{
    std::ifstream checkFile(fileName, std::ios::in | std::ios::binary);
    if (checkFile.fail())
        return false;
    return true;
}