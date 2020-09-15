#include "include/mcm/base/json/json.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"

static void generateJSONFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateJSON)
        .setFunc(generateJSONFcn)
        .defineArg(json::JSONType::String, "filename")
        .setDescription(
            "Saves the computation model as a JSON file, plus any populated tensors"
        );
    }

}

static void generateJSONFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Saving computation model to json...");

    // get filename from compilation descriptor
    std::string outputPath = "computation_model.json"; 
    if (passDesc.hasAttr("filename")) {
        outputPath = passDesc.get<std::string>("filename");
    }
    // use same filename for tensor outputs (with different extension)
    size_t lastindex = outputPath.find_last_of(".");
    std::string outputPathNoExt(outputPath.substr(0, lastindex));

    std::ofstream ostream;
    ostream.open(outputPath, std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw mv::ArgumentError(model, "filename", outputPath, "Unable to open output filename");

    mv::json::Value computationModel = model.toJSON();
    ostream << computationModel.stringifyPretty();
    ostream.close();

    //Any populated tensors are serialized in different files
    if(model.hasPopulatedTensorsToJSON())
    {
        for(auto tensorIt = model.tensorBegin(); tensorIt != model.tensorEnd(); ++tensorIt)
        {
            if(!tensorIt->isPopulated())
                continue;
            std::string currentTensorOutputPath(outputPathNoExt+"_"+tensorIt->getName());
            std::ofstream currentTensorOutputStream(currentTensorOutputPath, std::ios::trunc | std::ios::out | std::ios::binary);

            if (tensorIt->isDoubleType())
            {
                std::vector<double> tensorData(tensorIt->getDoubleData());
                currentTensorOutputStream.write(reinterpret_cast<char*>(&tensorData[0]), tensorData.size() * sizeof(tensorData[0]));
            }
            else
            {
                std::vector<int64_t> tensorData(tensorIt->getIntData());
                currentTensorOutputStream.write(reinterpret_cast<char*>(&tensorData[0]), tensorData.size() * sizeof(tensorData[0]));
            }
            currentTensorOutputStream.close();
        }
    }
}
