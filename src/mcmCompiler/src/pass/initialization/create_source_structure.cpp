#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"

static void CreateSourceStructureFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(CreateSourceStructure)
            .setFunc(CreateSourceStructureFcn)
            .setDescription(
                "Creates the source structure of the GraphFile");
    }
}

void CreateSourceStructureFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& compilationDescriptor, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(td);
    rm.buildHeader(model, compilationDescriptor);
}
