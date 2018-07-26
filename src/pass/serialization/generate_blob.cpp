#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"

static void generateBlobFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateBlob)
        .setFunc(generateBlobFcn)
        .setGenre(PassGenre::Serialization)
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates an executable blob file"
        );
        
    }

}

void generateBlobFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput)
{   

    using namespace mv;

    if (compDesc["GenerateBlob"]["output"].get<std::string>().empty())
        throw ArgumentError("output", "", "Unspecified output name for generate dot pass");

    mv::ControlModel cm(model);
    mv::Serializer serializer(mv::mvblob_mode);
    long long result = static_cast<long long>(serializer.serialize(cm, compDesc["GenerateBlob"]["output"].get<std::string>().c_str()));
    compOutput["blobSize"] = result;

}