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
static void placeReQuantizeDepthwiseBefore(mv::OpModel & om, mv::Data::OpListIterator opIt, std::size_t index, mv::Data::TensorIterator inputTensor, mv::QuantizationParams quantParams);

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

void preprocessForPWL(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    bool PWLUsage = globalParams->hasAttr("PWLUsage") ? globalParams->get<bool>("PWLUsage") : false;

    const std::vector<std::string> DPU_OPS = { "Conv", "FullyConnected", "DepthwiseConv", "Deconv", "MaxPool", };

    if (PWLUsage) {
        // NOTE: find convolutions that have lrelu / mish / etc as postOp
        auto sortedOps = om.topologicalSort();

        for (auto opIterator : sortedOps) {
            if (std::find(DPU_OPS.begin(), DPU_OPS.end(), opIterator->getOpType()) != DPU_OPS.end() &&
                opIterator->hasAttr("postOpTypes")) {
                auto postOpTypes = opIterator->get<std::vector<std::string>>("postOpTypes");
                auto dpuPostOp = std::find_if(postOpTypes.begin(), postOpTypes.end(), mv::ControlModel::isDpuPwl);
                if (dpuPostOp != postOpTypes.end()) {
                    opIterator->set<bool>("WithDPUPWL", true);
                    opIterator->set<bool>("With" + *dpuPostOp, true);
                }
            }
        }
    }
}

void checkPWLForRequantize(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto convOps = om.getOps("Conv");
    std::unordered_map<std::string, std::pair<int, int>> pwl_ranges = {
        {"LeakyRelu", {-128, 127}},
        {"Mish", {-128, 127}},
    };

    for (auto conv:convOps)
    {
        if (conv->hasAttr("postOpTypes"))
        {
            auto postOpTypes = conv->get<std::vector<std::string>>("postOpTypes");
            auto dpuPostOp = std::find_if(postOpTypes.begin(), postOpTypes.end(), mv::ControlModel::isDpuPwl);
            if (dpuPostOp != postOpTypes.end())
            {
                auto pwl_range = pwl_ranges[*dpuPostOp];

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
                if (*dpuPostOp == "Mish") {
                    const std::map<int32_t, std::pair<int, int>> MISH_RANGES = {
                        {65000, {-128, 127}},
                        {66757, {-128, 127}},
                        {73281, {-128, 127}},
                        {74804, {-128, 127}},
                        {76484, {-128, 127}},
                        {76718, {-128, 127}},
                        {81015, {-128, 127}},
                        {81484, {-128, 127}},
                        {81875, {-128, 127}},
                        {82656, {-128, 127}},
                        {82812, {-128, 127}},
                        {82813, {-128, 127}},
                        {86953, {-128, 127}},
                        {88437, {-128, 127}},
                        {89219, {-128, 127}},
                        {90000, {-128, 127}},
                        {91484, {-128, 127}},
                        {91875, {-128, 127}},
                        {92343, {-128, 127}},
                        {92344, {-128, 127}},
                        {93515, {-128, 127}},
                        {93516, {-128, 127}},
                        {94687, {-128, 127}},
                        {95781, {-128, 127}},
                        {96171, {-128, 127}},
                        {96172, {-128, 127}},
                        {96640, {-128, 127}},
                        {96641, {-128, 127}},
                        {97031, {-128, 127}},
                        {98281, {-128, 127}},
                        {98437, {-128, 127}},
                        {98438, {-128, 127}},
                        {98984, {-128, 127}},
                        {99140, {-128, 127}},
                        {99843, {-128, 127}},
                        {99844, {-128, 127}},
                        {101641, {-128, 127}},
                        {101875, {-128, 127}},
                        {102578, {-128, 127}},
                        {103280, {-128, 127}},
                        {107422, {-128, 127}},
                        {109453, {-128, 127}},
                        {112266, {-128, 127}},
                        {112500, {-128, 127}},
                        {113047, {-128, 127}},
                        {114375, {-128, 127}},
                        {116641, {-128, 127}},
                        {122655, {-128, 127}},
                        {124375, {-128, 127}},
                        {124766, {-128, 127}},
                        {131719, {-128, 127}},
                        {131797, {-128, 127}},
                        {132266, {-128, 127}},
                        {133281, {-128, 127}},
                        {150234, {-128, 127}},
                        {153047, {-128, 127}},
                        {153828, {-128, 127}},
                        {160938, {-128, 127}},
                        {161719, {-128, 127}},
                        {161875, {-128, 127}},
                        {164375, {-128, 127}},
                        {169531, {-128, 127}},
                        {178438, {-128, 127}},
                        {189061, {-128, 127}},
                        {192188, {-128, 127}},
                        {192656, {-128, 127}},
                        {193594, {-128, 127}},
                        {198281, {-128, 127}},
                        {198282, {-128, 127}},
                        {215156, {-128, 127}},
                        {237500, {-128, 127}},
                        {238438, {-128, 127}},
                        {254844, {-128, 127}},
                        {269531, {-128, 127}},
                        {307500, {-128, 127}},
                        {355312, {-128, 127}},
                        {355313, {-128, 127}},
                        {388125, {-128, 127}},

                        {112253, {-128, 127}},
                        {113965, {-128, 127}},
                        {91865, {-128, 127}},
                        {88441, {-128, 127}},
                        {133264, {-128, 127}},
                        {261082, {-128, 127}},
                        {228631, {-128, 127}},
                        {216725, {-128, 127}},
                        {101787, {-128, 127}},
                        {97935, {-128, 127}},
                        {94666, {-128, 127}},
                        {161863, {-128, 127}},
                        {285828, {-128, 127}},
                        {206375, {-128, 127}},
                        {205441, {-128, 127}},
                        {170423, {-128, 127}},
                        {137817, {-128, 127}},
                        {107429, {-128, 127}},
                        {130938, {-128, 127}},
                        {269564, {-128, 127}},
                        {102020, {-128, 127}},
                        {254856, {-128, 127}},
                        {232989, {-128, 127}},
                        {103265, {-128, 127}},
                        {192601, {-128, 127}},
                        {207231, {-128, 127}},
                        {123109, {-128, 127}},
                    };
                    int32_t max_quant = std::round(outQuantParams.getMax().at(0) * 10000.f);
                    if (MISH_RANGES.count(max_quant) > 0) {
                        pwl_range = MISH_RANGES.at(max_quant);
                    } else {
                        throw std::runtime_error("preprocess_PWL: Couldn't find max_quant: " + std::to_string(max_quant));
                    }
                }

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
}
