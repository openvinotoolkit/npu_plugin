#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <sys/stat.h>

void convertFlatbufferFcn(const mv::pass::PassEntry& pass, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ConvertFlatbuffer)
        .setFunc(convertFlatbufferFcn)
        .defineArg(json::JSONType::String, "input")
        .setLabel("Debug")
        .setDescription(
            "Converts a flatbuffer binary file in json under UNIX"
        );

    }

}

void convertFlatbufferFcn(const mv::pass::PassEntry& pass, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    std::string outputFile = passDesc.get<std::string>("input");
    struct stat buffer;
    if (stat (outputFile.c_str(), &buffer) == 0)
    {
        std::string flatbufferCommand("flatc -t " + mv::utils::projectRootPath() + "/schema/graphfile/src/schema/graphfile.fbs --strict-json -- " + outputFile);
        system(flatbufferCommand.c_str());
    }
    else
        pass.log(mv::Logger::MessageType::Error, outputFile + " not found");
}
