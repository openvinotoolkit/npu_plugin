#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include <sys/stat.h>

void convertDotFcn(const mv::pass::PassEntry& pass, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ConvertDot)
        .setFunc(convertDotFcn)
        .defineArg(json::JSONType::String, "input")
        .setLabel("Debug")
        .setDescription(
            "Converts a .dot file in SVG under UNIX"
        );

    }

}

void convertDotFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    auto globalConfigParams = model.getGlobalConfigParams();

    auto convert = globalConfigParams->hasAttr("ConvertDots") ? globalConfigParams->get<bool>("ConvertDots") : true;
    if (convert)
    {
        std::string outputFile = passDesc.get<std::string>("input");
        system(("dot -Tsvg " + outputFile + " -o " + outputFile+".svg").c_str());
    }
}
