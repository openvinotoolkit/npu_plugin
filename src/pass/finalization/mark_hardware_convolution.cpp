#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/base/json/number_float.hpp"

static void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void scaleFissionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&);
static void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MarkHardwareConvolution)
        .setFunc(markHardwareConvolution)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass marks the convolutions that can be executed in NCE"
        );

        MV_REGISTER_PASS(ScaleFission)
        .setFunc(scaleFissionFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "Adds scales around HW ops to utilize more bits of fixed-point number representation in MAC HW units"
        );

        MV_REGISTER_PASS(FormatMXWeights)
        .setFunc(formatMXWeights)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass reshapes relevant Convolution weights for the MyriadX NCE"
        );
    }
}

//NOTE: This should not be done in such hardcoded way.
void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{

    int amount_marked = 0;
    int mark_limit = 3;

    bool disableHardware = false;
    if (compDesc.hasKey("MarkHardwareConvolution"))
        if (compDesc["MarkHardwareConvolution"].hasKey("disableHardware"))
            if (compDesc["MarkHardwareConvolution"]["disableHardware"].get<bool>())
                disableHardware = true;

    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (!disableHardware)
        {
            if(!opIterator->isHardwarizeable(compDesc) || amount_marked >= mark_limit)
            {
                om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                continue;
            }

            om.addAttr(opIterator, "NCE1_Compatible", (int)1);
            om.addAttr(opIterator, "NCE1_AssignedCMX", (int)0);
            ++amount_marked;
        }
        else
            om.addAttr(opIterator, "NCE1_Compatible", (int)0);
    }
}

void scaleFissionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    std::string const PASS_NAME = "ScaleFission";
    std::string const FACTOR_KEY = "scalefactors";
    std::string opName = "";

    double upNum = 1.0;

    if (!compDesc.hasKey("pass"))
    {
        return ;
    }
    if (!compDesc["pass"].hasKey(PASS_NAME))
    {
        return ;
    }
    if (!compDesc["pass"][PASS_NAME].hasKey(FACTOR_KEY))
    {
        return ;
    }

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        opName =opIt->getName();
        if ((opIt->getOpType() == OpType::Conv2D)&&(opIt->hasAttr("NCE1_Compatible")))
        {
            if (opIt->get<int>("NCE1_Compatible") == 1)
            {
                if (compDesc["pass"][PASS_NAME][FACTOR_KEY].hasKey(opName))
                {
                    upNum = compDesc["pass"][PASS_NAME][FACTOR_KEY][opName].get<double>();

                    std::vector<double> scaleUpWData = mv::utils::generateSequence<double>(opIt->getInputTensor(1)->getShape().totalSize(), upNum, 0.0f);
                    std::vector<double> scaleDnData = mv::utils::generateSequence<double>(opIt->getOutputTensor(0)->getShape().totalSize(), (1.0f/upNum), 0.0f);

                    // scale (up) inputs by multiplying weights and bias
                    std::string scaleUpWTensorName = opName + "_scale_in";
                    auto scaleUpWeights = dm.defineTensor(scaleUpWTensorName, opIt->getInputTensor(1)->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, scaleUpWData);
                    opIt->getInputTensor(1)->multiply(*scaleUpWeights);

                    if (opIt->hasAttr("bias"))
                    {
                        auto biasTensor = dm.findTensor(opIt->get<std::string>("bias"));
                        std::vector<double> scaleUpBData = mv::utils::generateSequence<double>(biasTensor->getShape().totalSize(), upNum, 0.0f);
                        std::string scaleUpBTensorName = opName + "_scale_bias";
                        auto scaleUpBias = dm.defineTensor(scaleUpBTensorName, biasTensor->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, scaleUpBData);
                        biasTensor->multiply(*scaleUpBias);
                    }

                    // scale (down) output by adding HWscale attributes to conv
                    std::string scaleTensorName = opName + "_scale";
                    auto scaleTensor = dm.defineTensor(scaleTensorName, opIt->getOutputTensor(0)->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, scaleDnData);
                    Attribute scaleAttr(scaleTensor->getName());
                    om.addAttr(opIt, "scale", scaleAttr);
                }
                else
                {
                    upNum = 1.0;
                }
            }  // end HW conv
        }  // end conv
    }  // end op loop
}

//NOTE: This should not be done in such hardcoded way.
void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        bool valid = false;
        if(opIterator->hasAttr("NCE1_Compatible"))
        {
            valid = opIterator->get<int>("NCE1_Compatible");
        }
        if (valid){

            auto weights = opIterator->getInputTensor(1);
            auto wshape = weights->getShape();

            mv::Shape newShape = mv::Shape({wshape[3]/8, wshape[2], wshape[1], wshape[0], 8});

            mv::Tensor newTensor = mv::Tensor("MX_Weights",
                                                newShape,
                                                weights->getDType(),
                                                weights->getOrder());

            std::vector<double> new_data;
            auto data = weights->getData();

            unsigned int o_iC = wshape[2], o_oC = wshape[3], o_fw = wshape[1];

            for(std::size_t i = 0; i != newShape[0]; i++){
                for(std::size_t j = 0; j != newShape[1]; j++){
                    for(std::size_t x = 0; x != newShape[2]; x++){
                        for(std::size_t y = 0; y != newShape[3]; y++){
                            for(std::size_t z = 0; z != newShape[4]; z++){
                                new_data.push_back(data[
                                    x*o_fw*o_iC*o_oC +  // Kernel Height is largest Dim in original matrix.
                                    y*o_iC*o_oC +       // Followed by Width
                                    j*o_oC +            // then Input Channels
                                    i*8 + z             // Output Channels are written in blocks of 8
                                ]);
                            }
                        }
                    }
                }
            }

            newTensor.populate(new_data, weights->getOrder());

            auto new_op = om.constant(
                newTensor.getData(),
                newTensor.getShape(),
                newTensor.getDType(),
                newTensor.getOrder(),
                mv::OpType(mv::OpType::Constant).toString() + "_" + std::to_string(om.opsCount(mv::OpType::Constant)) + "MxWeights"
            );

            opIterator->setInputTensor(new_op, 1);

        }
    }
    std::cout << "exiting formatMXweights pass " << std::endl;
}
