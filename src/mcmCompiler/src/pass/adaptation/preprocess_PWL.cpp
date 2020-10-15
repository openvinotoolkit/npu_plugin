#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <functional>

static void preprocessForPWL(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void checkPWLForRequantize(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void placeReQuantizeDepthwiseBefore(mv::OpModel om, mv::Data::OpListIterator opIt, std::size_t index, mv::Data::TensorIterator inputTensor, mv::QuantizationParams quantParams);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(PreprocessForPWL)
        .setFunc(preprocessForPWL)
        .setDescription(
            "Preprocess appropriately the operations that have leaky Relu as a post Op for the PWL approach:\
            You need to mark the first guy cause it is going to load the registers, the rest need to take FLEXARB."
        );

        MV_REGISTER_PASS(CheckPWLForRequantize)
        .setFunc(checkPWLForRequantize)
        .setDescription(
            "The PWL has a constant quantization range that can function, so if there is quantization \
             out of this range you need to requantize to the constant one and then add a depthwise that will lead you back."
        );
    }
}

void placeReQuantizeDepthwiseBefore(mv::OpModel om, mv::Data::OpListIterator opIt, std::size_t index, mv::Data::TensorIterator inputTensor, mv::QuantizationParams quantParams)
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
    weights = om.constantInt(weightsData,
                        {1, 1, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
                        mv::DType("UInt8"),
                        mv::Order(mv::Order::getRowMajorID(4)),
                        weightsQuantParams);
    auto reQuantizeDepthwise = om.depthwiseConv(inputTensor, weights, {1,1}, {0, 0, 0, 0},
                        1, mv::DType("UInt8"), quantParams, opIt->getName() + "Depthwise_requant" + std::to_string(index));
    auto reQuantizeDepthwiseOp = om.getSourceOp(reQuantizeDepthwise);
    auto weightsOp = om.getSourceOp(weights);
    reQuantizeDepthwiseOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
    weightsOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
    om.undefineFlow(inputFlow);
    opIt->setInputTensor(reQuantizeDepthwise, index, false);
    om.defineFlow(reQuantizeDepthwise, opIt, index);
}

void preprocessForPWL(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    bool PWLUsage = globalParams->hasAttr("PWLUsage") ? globalParams->get<bool>("PWLUsage") : false;
    if (PWLUsage)
    {
        //NOTE: find the first convolution that has lrelu as postOp
        bool foundFirstConv = false;
        auto sortedOps = om.topologicalSort();
        mv::Data::OpListIterator firstConv;
        for (auto opIterator: sortedOps)
        {
            if (opIterator->getOpType() == "Conv" && opIterator->hasAttr("postOpTypes"))
            {
                auto postOpTypes = opIterator->get<std::vector<std::string>>("postOpTypes");
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "LeakyRelu") != postOpTypes.end())
                {
                    foundFirstConv = true;
                    firstConv = opIterator;
                    break;
                }
            }
        }
        //NOTE: opIterator contains the first conv with lrelu if no conv with lrelu just go to next pass
        if (foundFirstConv)
        {
            firstConv->set<bool>("firstConvWithLRelu", true);
            auto convOps = om.getOps("Conv");
            for (auto& convOp : convOps)
            {
                if (convOp->hasAttr("postOpTypes"))
                {
                    auto postOpTypes = convOp->get<std::vector<std::string>>("postOpTypes");
                    auto itr = std::find(postOpTypes.begin(), postOpTypes.end(), "LeakyRelu");
                    if (itr != postOpTypes.end())
                    {
                        replace(postOpTypes.begin(), postOpTypes.end(), +"LeakyRelu", +"FLEXARB");
                        convOp->set<std::vector<std::string>>("postOpTypes", postOpTypes);
                    }
                }
            }
        }
    }
    return;
}

void checkPWLForRequantize(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto convOps = om.getOps("Conv");
    std::pair<int, int> pwl_range = {-128, 127};
    for (auto conv:convOps)
    {
        if (conv->hasAttr("postOpTypes"))
        {
            auto postOpTypes = conv->get<std::vector<std::string>>("postOpTypes");
            if (std::find(postOpTypes.begin(), postOpTypes.end(), "FLEXARB") != postOpTypes.end())
            {
                //NOTE: the idea is that the pwl needs constant quantized range (-128,128) in order
                // to be maximum accurate so we apply the formula below, the idea is that we keep
                // same float range and zero point and we balance with changing scale, quantized range
                auto outQuantParams = conv->get<mv::QuantizationParams>("quantParams");
                double fl_min = outQuantParams.getMin()[0];
                double fl_min_before_lrelu = outQuantParams.getMin()[0]/conv->get<double>("leakyAlpha");
                double fl_max = outQuantParams.getMax()[0];
                int q_min = std::round(fl_min_before_lrelu/outQuantParams.getScale()[0] + outQuantParams.getZeroPoint()[0]);
                int q_max = std::round(fl_max/outQuantParams.getScale()[0] + outQuantParams.getZeroPoint()[0]);

                double last_fl_max = fl_max;
                double last_q_max = pwl_range.second;
                //NOTE: zero point is added after lr so the formula should contain the subtraction
                if(q_max - outQuantParams.getZeroPoint()[0] > pwl_range.second ||
                    std::round(std::abs(q_min) * conv->get<double>("leakyAlpha")) - outQuantParams.getZeroPoint()[0] > std::abs(pwl_range.first))
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
                    conv->set<mv::QuantizationParams>("quantParams", newQuantParams);
                    outputTensor->set<mv::QuantizationParams>("quantParams", newQuantParams);
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
    return;
}
