#include "include/mcm/pass/serialization/generate_blob.hpp"
#include "include/mcm/deployer/serializer.hpp"

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateBlob)
        .setFunc(__generate_blob_detail_::generateBlobFcn)
        .setGenre(PassGenre::Serialization)
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates an executable blob file"
        );
        
    }

}

void mv::pass::__generate_blob_detail_::generateBlobFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc)
{   

   __debug_generateBlobFcn_(model, targetDesc, compDesc);

}

uint64_t mv::pass::__generate_blob_detail_::__debug_generateBlobFcn_(ComputationModel& model, TargetDescriptor&, json::Object& compDesc)
{
    if (compDesc["GenerateBlob"]["output"].get<std::string>().empty())
        throw ArgumentError("output", "", "Unspecified output name for generate dot pass");

    mv::ControlModel cmresnet(model);
    mv::Serializer serializer(mv::mvblob_mode);
    return serializer.serialize(cmresnet, compDesc["GenerateBlob"]["output"].get<std::string>().c_str());
}