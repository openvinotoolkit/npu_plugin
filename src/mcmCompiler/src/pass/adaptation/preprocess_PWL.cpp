#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <functional>

static void checkPWLForRequantize(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void placeReQuantizeDepthwiseBefore(mv::OpModel & om, mv::Data::OpListIterator opIt, std::size_t index, mv::Data::TensorIterator inputTensor, mv::QuantizationParams quantParams);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(CheckPWLForRequantize)
        .setFunc(checkPWLForRequantize)
        .setDescription(
            "The PWL has a constant quantization range that can function, so if there is quantization \
             out of this range you need to requantize to the constant one and then add a depthwise that will lead you back."
        );
    }
}

void placeReQuantizeDepthwiseBefore(mv::OpModel & om, mv::Data::OpListIterator opIt, std::size_t index, mv::Data::TensorIterator inputTensor, mv::QuantizationParams quantParams)
{
    //FIND THE APPROPRIATE FLOW
    auto inputFlow = opIt.leftmostInput();
    while(inputFlow != om.flowEnd())
    {
        auto tensor = inputFlow->getTensor();
        if (tensor->getName() == inputTensor->getName())
        {
            break;
        }
        ++inputFlow;
    }
    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = { 0 };
    std::vector<double> min = { 1 };
    std::vector<double> max = { 1 };

    std::vector<double> scale(1, 1.0f);
    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    int64_t weightsValue = 1;
    std::vector<int64_t> weightsData(inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
    weights = om.constantInt("",
                        weightsData,
                        {1, 1, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
                        mv::DType("UInt8"),
                        mv::Order(mv::Order::getRowMajorID(4)));
    weights->setQuantParams(weightsQuantParams);
    auto reQuantizeDepthwise = om.depthwiseConv(opIt->getName() + "Depthwise_requant" + std::to_string(index),
                        inputTensor, weights,
                        {1,1}, {0, 0, 0, 0}, 1);
    reQuantizeDepthwise->setDType(mv::DType("UInt8"));
    reQuantizeDepthwise->setQuantParams(quantParams);
    auto reQuantizeDepthwiseOp = om.getSourceOp(reQuantizeDepthwise);
    auto weightsOp = om.getSourceOp(weights);
    reQuantizeDepthwiseOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
    weightsOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
    om.undefineFlow(inputFlow);
    opIt->setInputTensor(reQuantizeDepthwise, index, false);
    om.defineFlow(reQuantizeDepthwise, opIt, index);
}

void checkPWLForRequantize(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto convOps = om.getOps("Conv");

    for (auto conv:convOps)
    {
        if (conv->hasAttr("PWLSource"))
        {
            auto pwl_source = conv->get<std::string>("PWLSource");

            std::pair<int, int> pwl_range;
            if(pwl_source == "Generation")
            {
                pwl_range = std::make_pair(-128, 127);
            }
            else if(pwl_source == "TargetDescriptor")
            {
                if(!conv->hasAttr("PWLType"))
                {
                    throw mv::AttributeError(conv->getLogID(), " is missing attribute with key \"PWLType\".");
                }
                if(!conv->hasAttr("PWLIndex"))
                {
                    throw mv::AttributeError(conv->getLogID(), " is missing attribute with key \"PWLIndex\". (even though PWL table source is target descriptor)");
                }

                auto pwl_type = conv->get<mv::PWLTableType>("PWLType");
                int pwl_table_index = conv->get<int>("PWLIndex");

                auto pwl_table_map = td.generalTargetConfigs().pwlTables;

                if(pwl_table_map.find(pwl_type) == pwl_table_map.end())
                {
                    mv::AttributeError(conv->getLogID(), " has invalid attribute with key \"PWLType\".");
                }

                auto pwl_table_vec = pwl_table_map.at(pwl_type);

                if(pwl_table_index >= static_cast<int>(pwl_table_vec.size()))
                {
                    mv::AttributeError(conv->getLogID(), " has attribute with key \"PWLIndex\" out of bounds.");
                }
                pwl_range = pwl_table_vec[pwl_table_index].float_range;
            }

            //NOTE: the idea is that the pwl needs constant quantized range (-4096, 4095) in order
            // to be maximum accurate so we apply the formula below, the idea is that we keep
            // same float range and zero point and we balance with changing scale, quantized range
            auto outQuantParams = conv->getOutputTensor(0)->getQuantParams();
            if (outQuantParams.getMin().empty()) {
                throw mv::ArgumentError(model,  "outQuantParams.getMin()",  "is empty", conv->getName());
            }
            if (outQuantParams.getMax().empty()) {
                throw mv::ArgumentError(model,  "outQuantParams.getMax()",  "is empty", conv->getName());
            }

            double fl_min = outQuantParams.getMin()[0];
            double fl_max = outQuantParams.getMax()[0];
            double scale = outQuantParams.getScale(0);
            if (scale == 0.0) {
                throw mv::ArgumentError(model, "getScale(0)",  " == 0",  conv->getName());
            }
            int q_max = std::round(fl_max / outQuantParams.getScale(0) + outQuantParams.getZeroPoint(0));

            double last_fl_max = fl_max;
            double last_q_max = pwl_range.second;
            //NOTE: zero point is added after lr so the formula should contain the subtraction
            bool leakyReluCase = false;
            if(conv->hasAttr("leakyAlpha")) {
                double leakyAlpha = conv->get<double>("leakyAlpha");
                if (leakyAlpha == 0.0) {
                    throw mv::AttributeError(model, "leakyAlpha==0");
                }
                double fl_min_before_lrelu = outQuantParams.getMin()[0] / leakyAlpha;
                int q_min = std::round(fl_min_before_lrelu/outQuantParams.getScale(0) + outQuantParams.getZeroPoint(0));
                const auto scaled_q_min = std::round(std::abs(q_min) * leakyAlpha);
                leakyReluCase = (scaled_q_min - outQuantParams.getZeroPoint(0)) > std::abs(pwl_range.first);
            }

            if(q_max - outQuantParams.getZeroPoint(0) > pwl_range.second || leakyReluCase)
            {
                // Note: for some reason the results without the absolute maximum are more accurate or
                // at least closer to cpu
//                    if (std::abs(q_min) > q_max)
//                    {
//                        last_fl_max = fl_min_before_lrelu;
//                        last_q_max = pwl_range.first;
//                    }
                int64_t newZp = outQuantParams.getZeroPoint()[0];
                double newScale = last_fl_max/(last_q_max-newZp);

                mv::QuantizationParams newQuantParams = mv::QuantizationParams({newZp},{newScale},
                                {fl_min},{fl_max});
                auto outputTensor = conv->getOutputTensor()[0];
                outputTensor->setQuantParams(newQuantParams);
                auto nextOps = mv::findSinkLayers(dm, conv->getOutputTensor()[0]);
                for (auto nextOp:nextOps)
                {
                    std::size_t index = 0;
                    for (std::size_t idx = 0; idx < nextOp->getInputTensor().size(); idx ++)
                    {
                        if (nextOp->getInputTensor()[idx]->getName() == outputTensor->getName())
                        {
                            index = idx;
                            break;
                        }
                    }
                    placeReQuantizeDepthwiseBefore(om, nextOp, index, outputTensor, outQuantParams);
                }
            }
        }
    }
}
