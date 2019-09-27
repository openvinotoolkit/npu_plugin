#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/myriadx/nce1.hpp"
#include "include/mcm/target/myriadx/nce1_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"


static void splitsOverH(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(SplitsOverH)
        .setFunc(splitsOverH)
        .setDescription(
            "This pass handles splits over H for each HW CONV"
        );
    }
}

unsigned computeMaxLines(mv::Nce1& nce, mv::Data::OpListIterator operationIt)
{
    auto input_tensor = operationIt->getInputTensor(0);
    auto input_tensor_shape = input_tensor->getShape();
    auto input_width = input_tensor_shape[0];

    auto output_tensor = operationIt->getOutputTensor(0);
    auto output_tensor_shape = output_tensor->getShape();
    auto output_width = output_tensor_shape[0];
    auto output_height = output_tensor_shape[1];

    //This is probably wrong
    if(operationIt->hasAttr("NCE1_CMX2CMX"))
        return output_height; //or output_height

    //Assuming split over H is always possible from this point on
    unsigned max_output_channels_performed = (unsigned)operationIt->get("NCE1_MaxOutputChannelsPerformed").get<std::size_t>();
    std::cout << "Max output channels performed " << max_output_channels_performed << std::endl;
    if(operationIt->getOpType() == "Conv")
        return nce.computeMaxOutputLinesConvolution(output_width, max_output_channels_performed);
    else //Pooling
    {
        std::array<unsigned short, 4> padding = {0, 0, 0, 0};
        std::array<unsigned short, 2> kernel = operationIt->get<std::array<unsigned short, 2>>("kSize");
        if(operationIt->hasAttr("padding"))
            padding = operationIt->get<std::array<unsigned short, 4>>("padding");
        return nce.computeMaxOutputLinesPooling(input_width, max_output_channels_performed, padding, kernel);
    }
}

std::vector<mv::SplitOverHSolution> computeSplitsOverH(mv::Nce1& nce, mv::Data::OpListIterator convIterator, unsigned max_lines)
{
    mv::ConvolutionParameters param = mv::fillKernel2DOperationParameters(convIterator);
    return nce.computeSplitsOverH(param, max_lines);
}

std::vector<mv::SplitOverHSolution> computeSplitsOverH(mv::Nce1& nce, mv::Data::OpListIterator opIterator)
{
    unsigned max_lines = computeMaxLines(nce, opIterator);
    std::cout << "Max lines " << max_lines << std::endl;
    return computeSplitsOverH(nce, opIterator, max_lines);
}

//ASSUMPTION: This pass must be executed after the mode selection pass.
//REASON: Paddings (and possibly modes) for each HW operation are needed.
void splitsOverH(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::Nce1 nce;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
    {
        if(!operationIt->hasAttr("NCE1_Compatible"))
            continue;
        if(!operationIt->get<int>("NCE1_Compatible"))
            continue;

        if(operationIt->getOpType() == "Conv" ||
           operationIt->getOpType() == "AvgPool" ||
           operationIt->getOpType() == "MaxPool")
        {

            std::vector<mv::SplitOverHSolution> splits = computeSplitsOverH(nce, operationIt);
            unsigned splits_over_height = splits.size();


            for(unsigned i = 0; i < splits_over_height; ++i)
                std::cout << i << " - " << splits[i] << std::endl;

            operationIt->set<std::size_t>("NCE1_SplitsOverHeight", (std::size_t)splits_over_height);

            // Compute DescriptorsSplits
            size_t splits_over_input_channels;
            if(operationIt->hasAttr("NCE1_SplitsOverInputChannels"))
                splits_over_input_channels = operationIt->get<size_t>("NCE1_SplitsOverInputChannels");
            else
                splits_over_input_channels = 1;

            operationIt->set<size_t>("NCE1_SplitsOverInputChannels", splits_over_input_channels);
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

}
