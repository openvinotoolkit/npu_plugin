#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/model/types.hpp"

static void modeSelection(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ModeSelection)
        .setFunc(modeSelection)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass selects the appropriate mode for each convolution executable by NCE"
        );
    }
}

mv::ModeSelectionResult optimize_convolution_nce1(mv::Nce1& nce, mv::Data::OpListIterator convIterator)
{
    mv::ModeSelectionNode source;
    auto weigth_tensor = convIterator->getInputTensor(1);
    auto input_tensor = convIterator->getInputTensor(0);
    auto output_tensor = convIterator->getOutputTensor(0);

    auto kernel_dimensions = weigth_tensor->getShape();
    auto input_dimensions = input_tensor->getShape();
    auto output_dimensions = output_tensor->getShape();

    source.parameters.kernel_x = kernel_dimensions[0];
    source.parameters.kernel_y = kernel_dimensions[1];
    source.parameters.input_width = input_dimensions[0];
    source.parameters.input_height = input_dimensions[1];
    source.parameters.input_channels = input_dimensions[2];
    source.parameters.output_width = output_dimensions[0];
    source.parameters.output_height = output_dimensions[1];
    source.parameters.output_channels = output_dimensions[2];

    auto strides = convIterator->getAttr("stride").getContent<mv::UnsignedVector2D>();
    source.parameters.stride_x = strides.e0;
    source.parameters.stride_y = strides.e1;

    source.remaining_output_channels = source.parameters.output_channels;
    return nce.optimize_convolution(source);
}

//NOTE: This should not be done in such hardcoded way.
void modeSelection(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {

    }
}
