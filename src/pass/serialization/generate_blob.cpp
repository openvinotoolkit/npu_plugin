#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"

static void generateBlobFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);
static void PopulateSerialFieldsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);
static void writeSerialFieldsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);

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
        .defineArg(json::JSONType::Bool, "enableFileOutput")
        .defineArg(json::JSONType::Bool, "enableRAMOutput")
        .setDescription(
            "Generates an executable blob file"
        );

    }

}

void generateBlobFcn(mv::ComputationModel& model, mv::TargetDescriptor& td, mv::json::Object& compDesc, mv::json::Object& compOutput)
{

    using namespace mv;

    mv::ControlModel cm(model);
    mv::Serializer serializer(mv::mvblob_mode);

    // Set output parameters for this serialization from config JSON object
    // note: defaults from cm.RuntimeBinary constructor are disableRam , enableFile ,  mcmCompile.blob
    bool RAMEnable = false ;
    bool fileEnable = false ;
    std::string blobFileName = "mcmCompile.blob"; 
 
    if (compDesc["GenerateBlob"]["enableRAMOutput"].get<bool>())
    {
        RAMEnable = true ;
    }
    cm.getBinaryBuffer()->setRAMEnabled(RAMEnable) ;

    if (compDesc["GenerateBlob"]["enableFileOutput"].get<bool>())
    {
        fileEnable = true ;
    }
    cm.getBinaryBuffer()->setFileEnabled(fileEnable) ;

    if (!(compDesc["GenerateBlob"]["fileName"].get<std::string>().empty()))
    {
        blobFileName = compDesc["GenerateBlob"]["fileName"].get<std::string>() ;
    }
    cm.getBinaryBuffer()->setFileName(blobFileName) ;

    long long result = static_cast<long long>(serializer.serialize(model, td);
    compOutput["blobSize"] = result;

}
static void PopulateSerialFieldsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& ){
    mv::OpModel om(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt){
        std::cout << "Populating Serial fields for Op{" << opIt->getOpType().toString() << "}" << std::endl;
        opIt->gatherSerialFields();
    }
}
