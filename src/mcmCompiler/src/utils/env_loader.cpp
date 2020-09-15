#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include <iostream>
#include <fstream>
#include <sys/stat.h>


std::string mv::utils::projectRootPath()
{
    
    std::string path = PROJECT_DIR;
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

void mv::utils::validatePath(const std::string& filename)
{
    // Check we're not trying to write to root, eg, /blob.bin
    if (filename.find("/") == 0)
        throw mv::ArgumentError("mv::Runtime", "file:location", "invalid", "Cannot write to root. Check configuration");

    // Check we're not trying to write to a higher directory, eg, ../path/to/blob
    if (filename.find("..") != std::string::npos)
        throw mv::ArgumentError("mv::Runtime", "file:location", "invalid", "Cannot write to higher directory. Check configuration");

    // Check if we have a folder structure in the filename
    if (filename.find("/") != std::string::npos)
    {
        // Check if folder structure exists, create if necessary, eg, a/b/c/blob.bin
        std::size_t found = filename.find_last_of("/");
        std::string path = filename.substr(0,found) + "/";
        std::string file = filename.substr(found+1);

        struct stat info;
        if( stat( path.c_str(), &info ) != 0 )
        {   // folder structure does not exist - create
            mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeModel", "Not found. Creating " + path);

            size_t pos = 0;
            std::string this_dir;
            std::string all_dir;
            while ((pos = path.find("/")) != std::string::npos)
            {
                // loop creating each dir a/b/c/d/
                this_dir = path.substr(0, pos);
                all_dir += this_dir + "/";
                mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeModel", "Making:  " + all_dir);
                mkdir(all_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                path.erase(0, pos+1);
            }
        }
        else if( info.st_mode & S_IFDIR ) //folder exists
            mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeModel", "Output location found:  " + path);
        else // not a directory - unknown error (probably a symbolic link)
            throw mv::ArgumentError("mv::Runtime", "file:location", "invalid", "Cannot write to output location. Check configuration.");
    }
}
