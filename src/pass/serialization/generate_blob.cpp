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

        MV_REGISTER_PASS(WriteSerialFields)
        .setFunc(writeSerialFieldsFcn)
        .setGenre(PassGenre::Serialization)
        .setDescription(
            "Writes fields for serialization"
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
        std::cout << "Populating Serial fields for Op{" << opIt->getOpType().toString() << "}" << std::endl;
        opIt->gatherSerialFields();
    }
}

static void writeSerialFieldsFcn(mv::ComputationModel& model, mv::TargetDescriptor& td, mv::json::Object& compDesc, mv::json::Object& compOutput){
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::cout << "Writing Serial Fields for Op{" << opIt->getOpType().toString() << "}" <<std::endl;

        // Get the serialization instructions for op
        mv::Element e("serial_viewer");
        if (opIt->getOpType() == mv::OpType::Input
            || opIt->getOpType() == mv::OpType::Output
            || opIt->getOpType() == mv::OpType::Constant){
            continue;
        }else if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible")){
            e = td.getSerialDefinition(opIt->getOpType().toString(), "NCE1");
        }else{
            e = td.getSerialDefinition(opIt->getOpType().toString(), "MvTensor");
        }

        std::vector<std::string> serial_instructions = e.get<std::vector<std::string>>("serial_view");
        for(auto s = serial_instructions.begin(); s != serial_instructions.end(); ++s){
            std::string instruction = s->substr(0, s->find(':'));
            std::string name = s->substr(s->find(':')+1, s->size());
            if(instruction == "Attr"){
                auto retrieved_attr = opIt->get<unsigned>(name);
                std::cout << "Retrieved: " << name << ": " <<retrieved_attr << std::endl;
            }else if(instruction == "Tensor"){
                std::string inOrOut = name.substr(0, name.find(':'));
                std::string index = name.substr(name.find(':')+1, name.size());
                mv::Data::TensorIterator retrievedT;
                std::cout << "Orig: "<< name << " - " <<inOrOut << ", " << index << ";" << std::endl;
                if(inOrOut == "0"){
                    unsigned idx = stoi(index);
                    std::cout << "Testing..."<< std::endl;
                    if(opIt->hasInputDef(idx))
                        retrievedT = opIt->getInputTensor(idx);
                    else
                        retrievedT = dm.tensorEnd();
                }else{
                    unsigned idx = stoi(index);
                    retrievedT = opIt->getOutputTensor(idx);
                }
                if(retrievedT == dm.tensorEnd())
                    std::cout << "Retrieved NULL: " << ": " << std::endl;
                else
                    std::cout << "Retrieved Tensor: " << ": " << retrievedT->getName() << std::endl;

            }else{
                // throw mv::AttributeError(instruction, "Invalid Serialization Instruction");
            }
        }


    }

}
