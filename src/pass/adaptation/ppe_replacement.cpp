#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include <functional>
#include "include/mcm/pass/pass_utils.hpp"

static void provideAccuracyinPPEs(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static const uint8_t ADD_INPUT_FLOWS = 2;

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(ProvideAccuracyinPPEs)
        .setFunc(provideAccuracyinPPEs)
        .setDescription(
            "Replaces the PPEs that suffer from accuracy issues"
        );
    }
}

mv::Data::OpListIterator portDepthwise(mv::OpModel& om, mv::Data::TensorIterator inputTensor, mv::Data::OpListIterator task,
                        mv::QuantizationParams quantParams, int64_t zeroPoint, double scaleValue, int64_t weightsValue)
{
    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = {zeroPoint};
    std::vector<double> min = {-std::numeric_limits<double>::infinity()};
    std::vector<double> max = {std::numeric_limits<double>::infinity()};
    auto inputShape = inputTensor->getShape();
    std::string name = task->getName();
    std::vector<double> scale(1, scaleValue);

    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    std::vector<int64_t> weightsData(inputShape[mv::IO_CHANNEL_DIMENSION], weightsValue);
    weights = om.constantInt(weightsData,
                        {1, 1, inputShape[mv::IO_CHANNEL_DIMENSION], 1},
                        mv::DType("UInt8"),
                        mv::Order(mv::Order::getRowMajorID(4)),
                        weightsQuantParams);
    auto depthwise_conv = om.depthwiseConv(inputTensor, weights, {1,1}, {0,0,0,0}, 1,
                                           mv::DType("UInt8"), quantParams, name + std::to_string(scaleValue) + "_PPEDepthwiseConv");
    auto depthwise_convOp = om.getSourceOp(depthwise_conv);
    auto weightsOp = om.getSourceOp(weights);
    unsigned currentOpId = task->get<unsigned>("opId");
    weightsOp->set<unsigned>("opId", currentOpId);
    depthwise_convOp->set<unsigned>("opId", currentOpId);

    return depthwise_convOp;
}

mv::Data::OpListIterator portRelu(mv::OpModel& om, mv::Data::TensorIterator inputTensor, mv::Data::OpListIterator task, mv::QuantizationParams quantParams)
{
    std::string name = task->getName();
    auto relu0 = om.relu(inputTensor, mv::DType("UInt8"), quantParams, name + "RELU");
    auto relu_op = om.getSourceOp(relu0);
    unsigned currentOpId = task->get<unsigned>("opId");
    relu_op->set<unsigned>("opId", currentOpId);
    return relu_op;
}

mv::Data::OpListIterator portAdd(mv::OpModel& om, std::vector <mv::Data::TensorIterator> inputs ,mv::Data::OpListIterator task, mv::QuantizationParams quantParams)
{
    std::string name = task->getName();
    auto eltwise = om.eltwise(inputs, "Add", mv::DType("UInt8"), quantParams, name + "PPEeltwise");
    auto eltwise_op = om.getSourceOp(eltwise);
    unsigned currentOpId = task->get<unsigned>("opId");
    eltwise_op->set<unsigned>("opId", currentOpId);
    return eltwise_op;
}

static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
{
    std::vector<mv::Data::OpListIterator> sinkOperations;
    auto flowsNames = (tensor)->get<std::set<std::string>>("flows");
    for(auto flowName : flowsNames)
    {
        auto df = dataModel.getDataFlow(flowName);
        sinkOperations.push_back(df.sink());
    }
    return sinkOperations;
}

void provideAccuracyinPPEs(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    bool PPEAccuracy = globalParams->hasAttr("PPEAccuracy") ? globalParams->get<bool>("PPEAccuracy") : false;
    if (!PPEAccuracy)
        return;
    auto leakyRelus = om.getOps("LeakyRelu");
    auto leakyRelu = leakyRelus.begin();
    while (leakyRelu != leakyRelus.end())
    {
        auto leakyReluOp = *leakyRelu;
        auto parentOp = om.getSourceOp(leakyReluOp->getInputTensor(0));
        auto leakyInputTensor = leakyReluOp->getInputTensor(0);
        auto leakyOutputTensor = leakyReluOp->getOutputTensor(0);
        std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, leakyOutputTensor);
        auto leakyReluQuantParams = leakyReluOp->get<mv::QuantizationParams>("quantParams");

        mv::Data::OpListIterator relu0, depthwise0, depthwise1, relu1, depthwise2;
        for (uint8_t i = 0; i < ADD_INPUT_FLOWS; i++)
        {
            if (i == 0)
            {
                depthwise0 = portDepthwise(om, leakyInputTensor, leakyReluOp, leakyReluQuantParams, 0, 1.0, 1);
                relu0 = portRelu(om, depthwise0->getOutputTensor(0), depthwise0, leakyReluQuantParams);
            }
            else
            {
                depthwise1 = portDepthwise(om, leakyInputTensor, leakyReluOp, leakyReluQuantParams, 255, 0.00392156862745098, 0);
                relu1 = portRelu(om, depthwise1->getOutputTensor(0), depthwise1, leakyReluQuantParams);
                double_t scale = leakyReluOp->get<double>("alpha")/255.0;
                depthwise2 = portDepthwise(om, relu1->getOutputTensor(0), leakyReluOp, leakyReluQuantParams, 255, scale, 0);
            }
        }
        std::vector<mv::Data::TensorIterator> inputs;
        inputs.push_back(relu0->getOutputTensor(0));
        inputs.push_back(depthwise2->getOutputTensor(0));
        auto add0 = portAdd(om, inputs, leakyReluOp, leakyReluQuantParams);
        auto backup = parentOp.leftmostOutput();
        om.undefineFlow(backup);
        om.removeOp(leakyReluOp);
        sinkOperators[0]->setInputTensor(add0->getOutputTensor(0), 0, false);
        om.defineFlow(add0->getOutputTensor(0), sinkOperators[0], 0);
        leakyRelu++;
    }
}
