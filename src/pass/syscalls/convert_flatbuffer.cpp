#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include <sys/stat.h>

void convertFlatbufferFcn(const mv::pass::PassEntry& pass, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ConvertFlatbuffer)
        .setFunc(convertFlatbufferFcn)
        .defineArg(json::JSONType::String, "input")
        .setDescription(
            "Converts a flatbuffer binary file in json under UNIX"
        );

    }

}

void convertFlatbufferFcn(const mv::pass::PassEntry& pass, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    std::string outputFile = passDesc.get<std::string>("input");
    struct stat buffer;
    if (stat (outputFile.c_str(), &buffer) == 0)
    {
        std::string flatbufferCommand("flatc -t $MCM_HOME/schema/graphfile/src/schema/graphfile.fbs --strict-json -- " + outputFile);
        system(flatbufferCommand.c_str());
    }
    else
        pass.log(mv::Logger::MessageType::Error, outputFile + " not found");
}
