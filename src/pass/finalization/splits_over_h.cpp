#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/resource/nce1_utils.hpp"
#include "include/mcm/computation/model/types.hpp"

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
    return 10;
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

        unsigned max_lines = computeMaxLines(nce, operationIt);
        std::vector<mv::SplitOverHSolution> splits = computeSplitsOverH(nce, operationIt, max_lines);
        for(auto& split: splits)//TODO
        {
            split.input_lines_processed;
            split.output_lines_processed;
            split.junk_output_before;
            split.junk_output_after;
            split.start_input_line;
            split.end_input_line;
            split.start_output_line;
            split.end_output_line;
        }

    }

}
