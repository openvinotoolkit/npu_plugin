#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/resource/nce1_utils.hpp"
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void splitsOverH(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(SplitsOverH)
        .setFunc(splitsOverH)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass handles splits over H for each HW CONV"
        );
    }
}

unsigned computeMaxLines(mv::Nce1& nce, mv::Data::OpListIterator convIt)
{
    auto input_tensor = convIt->getInputTensor(0);
    auto input_tensor_shape = input_tensor->getShape();

    if(convIt->hasAttr("NCE1_CMX2CMX"))
        return input_tensor_shape[1];

    //Assuming split over H is always possible from this point on
    unsigned max_output_channels_performed = convIt->getAttr("NCE1_MaxOutputChannelsPerformed").getContent<unsigned>();
    return nce.computeMaxOutputLines(input_tensor_shape[0], max_output_channels_performed);
}

std::vector<mv::SplitOverHSolution> computeSplitsOverH(mv::Nce1& nce, mv::Data::OpListIterator convIterator, unsigned max_lines)
{
    mv::ConvolutionParameters param = mv::fillConvolutionParameters(convIterator);
    return nce.computeSplitsOverH(param, max_lines);
}

std::vector<mv::SplitOverHSolution> computeSplitsOverH(mv::Nce1& nce, mv::Data::OpListIterator convIterator)
{
    unsigned max_lines = computeMaxLines(nce, convIterator);
    return computeSplitsOverH(nce, convIterator, max_lines);
}

//ASSUMPTION: This pass must be executed after the mode selection pass.
//REASON: Paddings (and possibly modes) for each HW operation are needed.
void splitsOverH(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::Nce1 nce;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
    {
        if(operationIt->getOpType() != mv::OpType::Conv2D)
            continue;
        if(!operationIt->hasAttr("NCE1_Compatible"))
            continue;
        if(!operationIt->getAttr("NCE1_Compatible").getContent<int>())
            continue;

        std::vector<mv::SplitOverHSolution> splits = computeSplitsOverH(nce, operationIt);
        unsigned splits_over_height = splits.size();
        for(unsigned i = 0; i < splits_over_height; ++i)
            std::cout << splits[i] << std::endl;
        om.addAttr(operationIt, "NCE1_SplitsOverHeight", mv::Attribute(mv::AttrType::UnsignedType, splits_over_height));

        // Compute DescriptorsSplits
        unsigned splits_over_input_channels = operationIt->getAttr("NCE1_SplitsOverInputChannels").getContent<unsigned>();
        std::vector<unsigned> modes = operationIt->getAttr("NCE1_Modes").getContent<std::vector<unsigned>>();

        unsigned descriptor_splits = nce.computeDescriptorSplits(splits_over_height, splits_over_input_channels, modes.size());
        om.addAttr(operationIt, "NCE1_DescriptorSplits", mv::Attribute(mv::AttrType::UnsignedType, descriptor_splits));

        std::vector<unsigned> input_lines_processed(splits_over_height);
        std::vector<unsigned> output_lines_processed(splits_over_height);
        std::vector<unsigned> junk_output_before(splits_over_height);
        std::vector<unsigned> junk_output_after(splits_over_height);

        std::vector<unsigned> start_input_line(splits_over_height);
        std::vector<unsigned> end_input_line(splits_over_height);
        std::vector<unsigned> start_output_line(splits_over_height);
        std::vector<unsigned> end_output_line(splits_over_height);


        for(unsigned i = 0; i < splits_over_height; ++i)
        {
            auto split = splits[i];
            input_lines_processed[i] = split.input_lines_processed;
            output_lines_processed[i] = split.output_lines_processed;
            junk_output_before[i] = split.junk_output_before;
            junk_output_after[i] = split.junk_output_after;
            start_input_line[i] = split.start_input_line;
            end_input_line[i] = split.end_input_line;
            start_output_line[i] = split.start_output_line;
            end_output_line[i] = split.end_output_line;
        }

        om.addAttr(operationIt, "NCE1_InputLinesProcessed", mv::Attribute(mv::AttrType::UnsignedVecType, input_lines_processed));
        om.addAttr(operationIt, "NCE1_OutputLinesProcessed", mv::Attribute(mv::AttrType::UnsignedVecType, output_lines_processed));
        om.addAttr(operationIt, "NCE1_JunkOutputBefore", mv::Attribute(mv::AttrType::UnsignedVecType, junk_output_before));
        om.addAttr(operationIt, "NCE1_JunkOutputAfter", mv::Attribute(mv::AttrType::UnsignedVecType, junk_output_after));

        om.addAttr(operationIt, "NCE1_StartInputLine", mv::Attribute(mv::AttrType::UnsignedVecType, start_input_line));
        om.addAttr(operationIt, "NCE1_EndInputLine", mv::Attribute(mv::AttrType::UnsignedVecType, end_input_line));
        om.addAttr(operationIt, "NCE1_StartOutputLine", mv::Attribute(mv::AttrType::UnsignedVecType, start_output_line));
        om.addAttr(operationIt, "NCE1_EndOutputLine", mv::Attribute(mv::AttrType::UnsignedVecType, end_output_line));
    }

}
