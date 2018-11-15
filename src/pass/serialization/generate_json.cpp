#include "include/mcm/base/json/json.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"

/*void generateJSONFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateJSON)
        .setFunc(generateJSONFcn)
        .setGenre(PassGenre::Serialization)
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates the JSON representation of computation model"
        );

    }

}

void generateJSONFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{
    std::ofstream ostream;
    std::string outputPath(compDesc["GenerateJSON"]["output"].get<std::string>());
    size_t lastindex = outputPath.find_last_of(".");
    std::string outputPathNoExt(outputPath.substr(0, lastindex));

    ostream.open(outputPath, std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw mv::ArgumentError("output", outputPath, "Unable to open output file");

    mv::json::Value computationModel = model.toJsonValue();
    ostream << computationModel.stringifyPretty();
    ostream.close();

    //Populated tensors must be serialized in different files
    if(mv::Jsonable::constructBoolTypeFromJson(computationModel["has_populated_tensors"]))
    {
        for(auto tensorIt = model.tensorBegin(); tensorIt != model.tensorEnd(); ++tensorIt)
        {
            if(!tensorIt->isPopulated())
                continue;
            std::string currentTensorOutputPath(outputPathNoExt+"_"+tensorIt->getName());
            std::ofstream currentTensorOutputStream(currentTensorOutputPath, std::ios::trunc | std::ios::out | std::ios::binary);
            std::vector<double> tensorData(tensorIt->getData());
            currentTensorOutputStream.write(reinterpret_cast<char*>(&tensorData[0]), tensorData.size() * sizeof(tensorData[0]));
            currentTensorOutputStream.close();
        }
    }
}*/
