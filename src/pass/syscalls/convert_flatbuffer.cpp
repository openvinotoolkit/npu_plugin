#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

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

void convertFlatbufferFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    std::string outputFile = passDesc.get<std::string>("input");
    std::string flatbufferCommand("flatc -t $MCM_HOME/schema/graphfile/src/schema/graphfile.fbs --strict-json --defaults-json -- " + outputFile);
    system(flatbufferCommand.c_str());
}
