#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void formatMXWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(FormatMXWeights)
        .setFunc(formatMXWeightsFcn)
        .setDescription(
            "This pass reshapes relevant Convolution weights for the MyriadX NCE"
        );
    }
}

void formatMXWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::Element &, mv::json::Object &)
{

    using namespace mv;

    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Conv")
        {
            if(opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible"))
            {
                auto oldWeights = opIt->getInputTensor(1);
                auto oldWeightsOp = om.getSourceOp(oldWeights);
                auto wshape = oldWeights->getShape();

                if(oldWeights->hasAttr("NCE1_WeightTransformed") && oldWeights->get<bool>("NCE1_WeightTransformed"))
                    continue;

                auto padding = oldWeights->get<std::vector<size_t>>("NCE1_Paddings");

                auto original_output_channel = wshape[3];
                auto original_output_channel_padding = padding[3];
                auto original_input_channel_padding = padding[2];

                original_output_channel += original_output_channel_padding;
                unsigned floor_output_channel = mv::ceil_division(original_output_channel, 8);
                mv::Shape new_shape = mv::Shape({floor_output_channel, wshape[2] + original_input_channel_padding, wshape[0], wshape[1], 8});

                pass.log(Logger::MessageType::Info, "Changing weight shape from " + wshape.toString() + " to " + new_shape.toString());
                mv::Data::TensorIterator new_weights;
                if (oldWeights->isDoubleType())
                {
                    std::vector<double> new_data(new_shape.totalSize(), 0);
                    mv::Tensor backup_tensor("backup", new_shape, oldWeights->getDType(), mv::Order(mv::Order::getRowMajorID(new_shape.ndims())), new_data);
                    for(unsigned kx = 0; kx < wshape[0]; ++kx)
                        for(unsigned ky = 0; ky < wshape[1]; ++ky)
                            for(unsigned ic = 0; ic < wshape[2]; ++ic)
                                for(unsigned oc = 0; oc < wshape[3]; ++oc)
                                    backup_tensor.at({oc/8,ic,ky,kx,oc%8}) = oldWeights->at({kx, ky, ic, oc});

                    new_data = backup_tensor.getDoubleData();
                    new_weights = om.constant(
                        new_data,
                        new_shape,
                        backup_tensor.getDType(),
                        backup_tensor.getOrder(),
                        oldWeights->getName() + "_MxWeights"
                    );
                }
                else
                {
                    std::vector<int64_t> new_data(new_shape.totalSize(), 0);
                    mv::Tensor backup_tensor("backup", new_shape, oldWeights->getDType(), mv::Order(mv::Order::getRowMajorID(new_shape.ndims())), new_data);
                    for(unsigned kx = 0; kx < wshape[0]; ++kx)
                        for(unsigned ky = 0; ky < wshape[1]; ++ky)
                            for(unsigned ic = 0; ic < wshape[2]; ++ic)
                                for(unsigned oc = 0; oc < wshape[3]; ++oc)
                                    backup_tensor.at({oc/8,ic,ky,kx,oc%8}) = oldWeights->at({kx, ky, ic, oc});

                    new_data = backup_tensor.getIntData();
                    new_weights = om.constantInt(
                        new_data,
                        new_shape,
                        backup_tensor.getDType(),
                        backup_tensor.getOrder(),
                        oldWeights->getName() + "_MxWeights"
                    );

                }


                new_weights->set<bool>("NCE1_WeightTransformed", true);
                new_weights->set<std::vector<size_t>>("NCE1_Paddings",
                    {0,
                     0,
                     0,
                     0,
                     0
                    });

                om.removeOp(oldWeightsOp);
                om.defineFlow(new_weights, opIt, 1);

                opIt->setInputTensor(new_weights, 1, false);
            }
        }
    }
    std::cout << "Exiting FormatMX Weights Pass " << std::endl;
}
