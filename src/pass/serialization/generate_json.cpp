#include "include/mcm/pass/serialization/generate_json.hpp"
#include "include/mcm/base/json/json.hpp"

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateJson)
        .setFunc(__generate_json_detail_::generateJsonFcn)
        .setGenre({PassGenre::Validation, PassGenre::Serialization})
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates the json representation of computation model"
        );

    }

}

void mv::pass::__generate_json_detail_::generateJsonFcn(ComputationModel& model, TargetDescriptor&, json::Object& compDesc)
{
    std::ofstream ostream;
    ostream.open(compDesc["GenerateJson"]["output"].get<std::string>(), std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw ArgumentError("output", compDesc["GenerateJson"]["output"].get<std::string>(), "Unable to open output file");

    mv::json::Value computationModel = model.toJsonValue();
    //NOTE: must become stringifypretty somehow
    ostream << computationModel.stringify();
}
