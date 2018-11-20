#include "include/mcm/utils/env_loader.hpp"
#include <iostream>

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
