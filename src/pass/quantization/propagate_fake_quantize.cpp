#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <math.h>

static void propagateParametersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(FakeQuantize)
        .setFunc(propagateParametersFcn)
        .setDescription(
            "Propagate quantization parametets from FakeQuantize layer"
        );
    }
}

mv::QuantizationParams extractQuantParams(mv::Data::OpListIterator fqOp) {
    assert(fqOp->getOpType() == "FakeQuantize");

    auto inputs = fqOp->getInputTensor();

    for (size_t i = 1; i < inputs.size(); ++i) {
        // TODO: pack tensor content to vectors
    }

    return mv::QuantizationParams{{0}, {2.0}, {}, {}};
}

void propagateParametersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&) {
    std::cout << "PASS CALLED\n";
    mv::OpModel om(model);

    auto fqOps = om.getOps("FakeQuantize");

    for (auto& fq : fqOps) {
        auto parent = om.getSourceOp(fq->getInputTensor(0));
        // parent->set<mv::QuantizationParams>("quantParams", extractQuantParams(fq));

        //TODO: this function doesn't work correct for weights because in mcm Weight is const tensor and
        // this functions removes all const inputs of the fq op.
        // TODO: Remove fix that was made to prevent this behaviourd
        linkNewOperationsRemove(parent, parent->getOutputTensor(0), om, fq);
    }
}
