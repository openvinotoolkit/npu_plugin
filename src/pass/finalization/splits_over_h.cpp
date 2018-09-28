#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/resource/nce1_utils.hpp"
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
    unsigned max_output_channels_performed = (unsigned)convIt->get("NCE1_MaxOutputChannelsPerformed").get<std::size_t>();
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
    mv::SplitOverHSolution a;
    a.input_lines_processed = 112; //224;
    a.output_lines_processed = 112; //224;
    a.input_lines_processed = 224;
    a.output_lines_processed = 112;
    a.junk_output_before = 0;
    a.junk_output_after =0;
    a.start_input_line = 0;
    a.end_input_line = a.input_lines_processed;
    a.start_output_line = 0;
    a.end_output_line = a.output_lines_processed;

    // mv::SplitOverHSolution b;
    // b.input_lines_processed = 112;
    // b.output_lines_processed = 112;
    // b.junk_output_before = 0;
    // b.junk_output_after = 0;
    // b.start_input_line = a.input_lines_processed;
    // b.end_input_line = a.input_lines_processed + b.input_lines_processed;
    // b.start_output_line = a.output_lines_processed;
    // b.end_output_line = a.output_lines_processed + b.output_lines_processed;

    // // std::vector<mv::SplitOverHSolution> g = {a, b};
    std::vector<mv::SplitOverHSolution> g = {a};
    return g;
    return computeSplitsOverH(nce, convIterator, max_lines);
}

//ASSUMPTION: This pass must be executed after the mode selection pass.
//REASON: Paddings (and possibly modes) for each HW operation are needed.
void splitsOverH(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    std::cout << "soh " << std::endl;
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::Nce1 nce;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
    {
        if(operationIt->getOpType() != mv::OpType::Conv2D)
            continue;
        if(!operationIt->hasAttr("NCE1_Compatible"))
            continue;
        if(!operationIt->get<int>("NCE1_Compatible"))
            continue;

        std::vector<mv::SplitOverHSolution> splits = computeSplitsOverH(nce, operationIt);
        unsigned splits_over_height = splits.size();

        /*
        for(unsigned i = 0; i < splits_over_height; ++i)
            std::cout << splits[i] << std::endl;
        */
        operationIt->set<std::size_t>("NCE1_SplitsOverHeight", (std::size_t)splits_over_height);

        // Compute DescriptorsSplits
        size_t splits_over_input_channels = operationIt->get<size_t>("NCE1_SplitsOverInputChannels");
        std::vector<size_t> modes = operationIt->get("NCE1_Modes").get<std::vector<std::size_t>>();

        unsigned descriptor_splits = nce.computeDescriptorSplits(splits_over_height, splits_over_input_channels, modes.size());
        operationIt->set<std::size_t>("NCE1_DescriptorSplits", (std::size_t)descriptor_splits);

        std::vector<std::size_t> input_lines_processed(splits_over_height);
        std::vector<std::size_t> output_lines_processed(splits_over_height);
        std::vector<std::size_t> junk_output_before(splits_over_height);
        std::vector<std::size_t> junk_output_after(splits_over_height);

        std::vector<std::size_t> start_input_line(splits_over_height);
        std::vector<std::size_t> end_input_line(splits_over_height);
        std::vector<std::size_t> start_output_line(splits_over_height);
        std::vector<std::size_t> end_output_line(splits_over_height);


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

        operationIt->set<std::vector<std::size_t>>("NCE1_InputLinesProcessed", input_lines_processed);
        operationIt->set<std::vector<std::size_t>>("NCE1_OutputLinesProcessed", output_lines_processed);
        operationIt->set<std::vector<std::size_t>>("NCE1_JunkOutputBefore", junk_output_before);
        operationIt->set<std::vector<std::size_t>>("NCE1_JunkOutputAfter", junk_output_after);

        operationIt->set<std::vector<std::size_t>>("NCE1_StartInputLine", start_input_line);
        operationIt->set<std::vector<std::size_t>>("NCE1_EndInputLine", end_input_line);
        operationIt->set<std::vector<std::size_t>>("NCE1_StartOutputLine", start_output_line);
        operationIt->set<std::vector<std::size_t>>("NCE1_EndOutputLine", end_output_line);
    }

}
