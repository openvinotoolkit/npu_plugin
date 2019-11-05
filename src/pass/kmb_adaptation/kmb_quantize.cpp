#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"


static void kmbQuantizeConversionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(KMBQuantizeConversion)
        .setFunc(kmbQuantizeConversionFcn)
        .setDescription(
            "This pass inserts Quantize conversion layers between DPUTask-to-UPATask transitions (& vice-versa)."
        );

    }

}


static void kmbQuantizeConversionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // Get current max opId
    unsigned currentId = 0;

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        currentId = std::max(currentId, opIt->get<unsigned>("opId"));
    }

    for (auto sink = om.opBegin(); sink != om.opEnd(); ++sink)
    {

        unsigned n = sink->inputSlots();
        for(unsigned i = 0; i < n; ++i)
        {
            auto sinkInputTensor = sink->getInputTensor(i);
            auto source = om.getSourceOp(sinkInputTensor);

            auto sourceType = source->getOpType();
            auto sinkType = sink->getOpType();

            if (sourceType == "Input" || sourceType == "Constant" || sourceType == "ConstantInt" || sourceType == "ConstantDataElement")
                continue;

            if (sinkType == "Output")
                continue;

            // HW-to-SW
            if ((sourceType != "UPATask") && (sinkType == "UPATask"))
            {
                auto taskOp = sink->get<std::string>("taskOp");
                if (taskOp != "Quantize")
                {
                    auto sinkOutputTensor = sink->getOutputTensor(0);

                    auto sinkDatatype = sink->get<mv::DType>("dType");
                    auto quantParams = source->get<mv::QuantizationParams>("quantParams");
                    //TODO: fix this; for UPATasks, temporarily assume "Default" is "Float16"
                    if (sinkDatatype == mv::DType("Default"))
                        sinkDatatype = mv::DType("Float16");

                    auto newOpOutputTensor = om.uPATaskQuantize({sinkInputTensor}, sinkDatatype, quantParams);

                    auto newOp = om.getSourceOp(newOpOutputTensor);
                    newOp->set<unsigned>("opId", currentId);
                    currentId++;

                    om.defineFlow(newOpOutputTensor, sink, 0);
                    om.undefineFlow(source.leftmostOutput());
                    sink->setInputTensor(newOpOutputTensor, 0, false);

                    // Copy quant params from source
                    sink->set<mv::QuantizationParams>("quantParams", quantParams);

                    // Set datatype of newOp output
                    newOpOutputTensor->set<mv::DType>("dType", sinkDatatype);
                }
            }
            // SW-to-HW
            else if ((sourceType == "UPATask") && (sinkType != "UPATask"))
            {
                auto taskOp = source->get<std::string>("taskOp");
                if (taskOp != "Quantize")
                {
                    auto sinkDatatype = sink->get<mv::DType>("dType");
                    auto quantParams = sink->get<mv::QuantizationParams>("quantParams");
                    //TODO: fix this; for DPUTasks, temporarily assume "Default" is "UInt8"
                    if (sinkDatatype == mv::DType("Default"))
                        sinkDatatype = mv::DType("UInt8");

                    auto newOpOutputTensor = om.uPATaskQuantize({sinkInputTensor}, sinkDatatype, quantParams);

                    auto newOp = om.getSourceOp(newOpOutputTensor);
                    newOp->set<unsigned>("opId", currentId);
                    currentId++;

                    om.defineFlow(newOpOutputTensor, sink, 0);
                    om.undefineFlow(source.leftmostOutput());
                    sink->setInputTensor(newOpOutputTensor, 0, false);

                    // Copy quant params from sink
                    source->set<mv::QuantizationParams>("quantParams", quantParams);

                    // Set datatype of newOp output
                    newOpOutputTensor->set<mv::DType>("dType", sinkDatatype);
                }
            }
        }
    }
}
