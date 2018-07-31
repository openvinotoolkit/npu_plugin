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
    std::string outputPath(compDesc["GenerateJson"]["output"].get<std::string>());
    size_t lastindex = outputPath.find_last_of(".");
    std::string outputPathNoExt(outputPath.substr(0, lastindex));

    ostream.open(outputPath, std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw mv::ArgumentError("output", outputPath, "Unable to open output file");

    mv::json::Value computationModel = model.toJsonValue();
    //NOTE: should become stringifypretty somehow
    ostream << computationModel.stringify();
    ostream.close();

    //Populated tensors must be serialized in different files
    if(mv::Jsonable::constructBoolTypeFromJson(computationModel["has_populated_tensors"]))
    {
        for(auto tensorIt = model.tensorBegin(); tensorIt != model.tensorEnd(); ++tensorIt)
        {
            std::string currentTensorOutputPath(outputPathNoExt+"_"+tensorIt->getName());
            std::ofstream currentTensorOutputStream(currentTensorOutputPath, std::ios::trunc | std::ios::out);
            mv::json::Value toDump = mv::Jsonable::toJsonValue(tensorIt->getData());
            currentTensorOutputStream << toDump.stringify();
            currentTensorOutputStream.close();
        }
    }
}
