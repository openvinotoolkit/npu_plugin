#include "include/mcm/base/json/json.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"

void generateJsonFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateJson)
        .setFunc(generateJsonFcn)
        .setGenre({PassGenre::Validation, PassGenre::Serialization})
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates the JSON representation of computation model"
        );

    }

}

void generateJsonFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{
    std::ofstream ostream;
    ostream.open(compDesc["GenerateJson"]["output"].get<std::string>(), std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw mv::ArgumentError("output", compDesc["GenerateJson"]["output"].get<std::string>(), "Unable to open output file");

    mv::json::Value computationModel = model.toJsonValue();
    //NOTE: must become stringifypretty somehow
    ostream << computationModel.stringify();
}
