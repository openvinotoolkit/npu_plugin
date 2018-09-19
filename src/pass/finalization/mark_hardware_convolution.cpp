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
void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{

    int amount_marked = 0;
    int mark_limit = 3;

    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if(!opIterator->isHardwarizeable(pobj) || amount_marked >= mark_limit)
        {
            om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
            continue;
        }

        om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 1));
        om.addAttr(opIterator, "NCE1_AssignedCMX", mv::Attribute(mv::AttrType::IntegerType, 0));
        ++amount_marked;        
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

    float upNum = 1.0f;

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
            if (opIt->getAttr("NCE1_Compatible").getContent<int>()==1)
            {
                if (compDesc["pass"][PASS_NAME][FACTOR_KEY].hasKey(opName))
                {
                    upNum = compDesc["pass"][PASS_NAME][FACTOR_KEY][opName].get<float>();

                    mv::dynamic_vector<mv::float_type> scaleUpWData = mv::utils::generateSequence<mv::float_type>(opIt->getInputTensor(1)->getShape().totalSize(), upNum, 0.0f);
                    mv::dynamic_vector<mv::float_type> scaleDnData = mv::utils::generateSequence<mv::float_type>(opIt->getOutputTensor(0)->getShape().totalSize(), (1.0f/upNum), 0.0f);

                    // scale (up) inputs by multiplying weights and bias
                    std::string scaleUpWTensorName = opName + "_scale_in";
                    auto scaleUpWeights = dm.defineTensor(scaleUpWTensorName, opIt->getInputTensor(1)->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, scaleUpWData);
                    opIt->getInputTensor(1)->multiply(*scaleUpWeights);

                    if (opIt->hasAttr("bias"))
                    {
                        auto biasTensor = dm.findTensor(opIt->getAttr("bias").getContent<std::string>());
                        mv::dynamic_vector<mv::float_type> scaleUpBData = mv::utils::generateSequence(biasTensor->getShape().totalSize(), upNum, 0.0f);
                        std::string scaleUpBTensorName = opName + "_scale_bias";
                        auto scaleUpBias = dm.defineTensor(scaleUpBTensorName, biasTensor->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, scaleUpBData);
                        biasTensor->multiply(*scaleUpBias);
                    }

                    // scale (down) output by adding HWscale attributes to conv
                    std::string scaleTensorName = opName + "_scale";
                    auto scaleTensor = dm.defineTensor(scaleTensorName, opIt->getOutputTensor(0)->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, scaleDnData);
                    Attribute scaleAttr(AttrType::StringType, scaleTensor->getName());
                    om.addAttr(opIt, "scale", scaleAttr);
                }
                else
                {
                    upNum = 1.0f;
                }
            }  // end HW conv
        }  // end conv
    }  // end op loop
}

//NOTE: This should not be done in such hardcoded way.
void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        bool valid = false;
        if(opIterator->hasAttr("NCE1_Compatible"))
        {
            valid = opIterator->getAttr("NCE1_Compatible").getContent<int>();
        }
        if (valid){

            auto weights = opIterator->getInputTensor(1);
            auto wshape = weights->getShape();

            std::cout << mv::Printable::toString(wshape) << std::endl;
            std::cout << wshape[3]/8 << ",64,1,1,8"<<std::endl;

            mv::Shape newShape = mv::Shape(
                // (mv::dim_type)wshape[0],
                // (mv::dim_type)wshape[1],
                // (mv::dim_type)(wshape[2] * wshape[3]/8),
                // (mv::dim_type)8
                32,
                64,
                1,
                1,
                8
            );

            mv::Tensor newTensor = mv::Tensor("MX_Weights",
                                                newShape,
                                                weights->getDType(),
                                                weights->getOrder());

            mv::dynamic_vector<mv::float_type> new_data;
            auto data = weights->getData();

            unsigned int o_iC = wshape[2], o_oC = wshape[3], o_fh = wshape[0], o_fw = wshape[1];

            for(int i = 0; i != newShape[0]; i++){
                for(int j = 0; j != newShape[1]; j++){
                    for(int x = 0; x != newShape[2]; x++){
                        for(int y = 0; y != newShape[3]; y++){
                            for(int z = 0; z != newShape[4]; z++){
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

            newTensor.populate(new_data);

            auto new_op = om.constant(
                newTensor.getData(),
                newTensor.getShape(),
                newTensor.getDType(),
                newTensor.getOrder(),
                mv::Printable::toString(mv::OpType::Constant) + "_" + mv::Printable::toString(om.opsCount(mv::OpType::Constant)) + "MxWeights"
            );

            opIterator->setInputTensor(new_op, 1);

        }
    }
std::cout << "exiting formatMXweights pass " << std::endl;
}
