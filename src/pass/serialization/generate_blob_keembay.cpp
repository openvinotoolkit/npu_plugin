#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/target/keembay/runtime_model/runtime_model.hpp"
#include "contrib/flatbuffers/include/flatbuffers/util.h"


static void generateBlobKeembayFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&td, mv::Element& passDesc, mv::json::Object& compOutput);
static void validatePath(const std::string& filename);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(GenerateBlobKeembay)
        .setFunc(generateBlobKeembayFcn)
        .setDescription(
            "Generates an executable blob file for Keembay"
        );

    }

}

void generateBlobKeembayFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::json::Object& compOutput)
{   
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance();
    rm.buildGraphFile(model, passDesc);

    if (!passDesc.hasAttr("output"))
        return;

    auto output = passDesc.get<std::string>("output");
    validatePath(output);

    rm.serialize(output);
}

void validatePath(const std::string& filename)
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
            while ((pos = path.find("/")) != std::string::npos) { 
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
