#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "include/mcm/utils/env_loader.hpp"

static void generateBlobKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&td, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(GenerateBlobKmb)
        .setFunc(generateBlobKmbFcn)
        .setDescription(
            "Generates an executable blob file for Kmb"
        );

    }

}

void generateBlobKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&)
{   

    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE)
    auto returnedParams = model.getGlobalConfigParams();
    auto huffmanCompression = returnedParams->get<bool>("HuffmanCompression");
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(huffmanCompression, td);
    rm.buildGraphFile(model, passDesc);

    if (!passDesc.hasAttr("output"))
        return;

    auto output = passDesc.get<std::string>("output");
    mv::utils::validatePath(output);

    rm.serialize(output);

}
