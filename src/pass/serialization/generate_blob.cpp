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
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates an executable blob file"
        );

        MV_REGISTER_PASS(writeSerialFields)
        .setFunc(writeSerialFieldsFcn)
        .setGenre(PassGenre::Serialization)
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Writes fields for serialization"
        );

    }

}

void generateBlobFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput)
{

    using namespace mv;

    if (compDesc["GenerateBlob"]["output"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate dot pass");

    mv::ControlModel cm(model);
    mv::Serializer serializer(mv::mvblob_mode);
    long long result = static_cast<long long>(serializer.serialize(cm, compDesc["GenerateBlob"]["output"].get<std::string>().c_str()));
    compOutput["blobSize"] = result;

}
static void PopulateSerialFieldsFcn(mv::ComputationModel& model, mv::TargetDescriptor& td, mv::json::Object& compDesc, mv::json::Object& compOutput){
    mv::OpModel om(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt){
        std::cout << "Op--" << opIt->getOpType().toString() <<std::endl;

        opIt->gatherSerialFields();
    }
}

static void writeSerialFieldsFcn(mv::ComputationModel& model, mv::TargetDescriptor& td, mv::json::Object& compDesc, mv::json::Object& compOutput){
    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::cout << "Op=" << opIterator->getOpType().toString() <<std::endl;
        // Get the serialization instructions for op
        mv::Element e("serial_viewer");
        if (opIterator->getOpType() == mv::OpType::Input
            || opIterator->getOpType() == mv::OpType::Output
            || opIterator->getOpType() == mv::OpType::Constant){
            continue;
        }else if (opIterator->hasAttr("NCE1_Compatible") && opIterator->get<int>("NCE1_Compatible")){
            e = td.getSerialDefinition(opIterator->getOpType().toString(), "NCE1");
        }else{
            e = td.getSerialDefinition(opIterator->getOpType().toString(), "MvTensor");
        }

        // std::cout << "Element: " << e.toString() << " - " << e.attrsCount()<< std::endl;

        std::vector<std::string> serial_instructions = e.get<std::vector<std::string>>("serial_view");
        for(auto s = serial_instructions.begin(); s != serial_instructions.end(); ++s){
            // std::cout << "Instruction " << *s << std::endl;
            std::string instruction = s->substr(0, s->find(':'));
            std::string name = s->substr(s->find(':')+1, s->size());
            if(instruction == "Attr"){
                opIterator->set<unsigned>("streamingMask", 1);
                std::cout << "Retrieved: " << name << ": " <<opIterator->get<unsigned>(name) << std::endl;
            }else if(instruction == "Tensor"){

            }else{
                // throw mv::AttributeError(instruction, "Invalid Serialization Instruction");
            }
        }


    }

}
