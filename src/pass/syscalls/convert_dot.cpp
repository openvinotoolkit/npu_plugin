#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

void convertDotFcn(const mv::pass::PassEntry& pass, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ConvertDot)
        .setFunc(convertDotFcn)
        .defineArg(json::JSONType::String, "input")
        .setDescription(
            "Converts a .dot file in SVG under UNIX"
        );

    }

}

void convertDotFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    std::string outputFile = passDesc.get<std::string>("input");
    system(("dot -Tsvg " + outputFile + " -o " + outputFile+".svg").c_str());
}
