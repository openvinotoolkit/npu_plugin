#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"

static void generateBlobFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&td, mv::json::Object& compDesc, mv::json::Object& compOutput);
static void PopulateSerialFieldsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object& compOutput);
//static void writeSerialFieldsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);

namespace mv
{

    namespace pass
    {


        MV_REGISTER_PASS(PopulateSerialFields)
        .setFunc(PopulateSerialFieldsFcn)
        .setGenre(PassGenre::Serialization)
        .setDescription(
            "Gathers fields for serialization"
        );

        MV_REGISTER_PASS(GenerateBlob)
        .setFunc(generateBlobFcn)
        .setGenre(PassGenre::Serialization)
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates an executable blob file"
        );

    }

}

void generateBlobFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::json::Object& compDesc, mv::json::Object& compOutput)
{   

    using namespace mv;

    if (compDesc["GenerateBlob"]["output"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate dot pass");

    mv::Serializer serializer(mv::mvblob_mode);
    long long result = static_cast<long long>(serializer.serialize(model, td, compDesc["GenerateBlob"]["output"].get<std::string>().c_str()));
    compOutput["blobSize"] = result;

}
void PopulateSerialFieldsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object& )
{
    mv::OpModel om(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::cout << "Populating Serial fields for Op{" << opIt->getOpType() << "}" << std::endl;
        //Short term fix: Big if-else acting like a switch
        //Long term solution:
    }
}
