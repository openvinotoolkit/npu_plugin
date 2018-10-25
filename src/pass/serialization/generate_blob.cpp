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
//        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates an executable blob file"
        );
        
    }

}

void generateBlobFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput)
{   

    using namespace mv;
    mv::ControlModel cm(model);
    mv::Serializer serializer(mv::mvblob_mode);

    std::cout << "in serilize pass, using RAM buffer" << std::endl;
    std::cout << " buffer size is " << cm.getBinarySize() << std::endl;
 
    cm.getBinaryBuffer("namedInPass",2000000000);
    std::cout << " after getBuffer size is " << cm.getBinarySize() << std::endl;

    long long result = static_cast<long long>(serializer.serialize(cm, compDesc["GenerateBlob"]["output"].get<std::string>().c_str()));
    compOutput["blobSize"] = result;

}
