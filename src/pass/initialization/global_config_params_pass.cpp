#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include <math.h>

static void GlobalConfigParamsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GlobalConfigParams)
        .setFunc(GlobalConfigParamsFcn)
        .setDescription(
            "Parses global config parameters from the Compilation Descriptor and stores them in the Computation Model."
        );
    }
}

static void GlobalConfigParamsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::json::Object&)
{
    //set the global params to be this pass's compilation descriptor element
    model.setGlobalConfigParams(compilationDescriptor);

    if (compilationDescriptor.hasAttr("verbose"))
    {
        auto verboseSetting = compilationDescriptor.get<std::string>("verbose");
        if (verboseSetting == "Error")
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Error);
        else if (verboseSetting == "Warning")
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Warning);
        else if (verboseSetting == "Info")
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
        else if (verboseSetting == "Debug")
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
        else
            throw mv::ArgumentError(pass, "CompilationDescriptor:GlobalConfigParams:verbose",
                verboseSetting, "Illegal value, legal values - Error, Warning, Info, Debug");
                
    }

}
