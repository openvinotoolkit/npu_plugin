#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(FormatMXWeights)
        .setFunc(formatMXWeights)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass reshapes relevant Convolution weights for the MyriadX NCE"
        );
    }
}


//NOTE: This should not be done in such hardcoded way.
void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto flowIt = dm.flowBegin();
    while (flowIt != dm.flowEnd())
    {
        auto source = flowIt.source();
        auto sink = flowIt.sink();

        bool valid = false;
        if(sink->hasAttr("NCE1_Compatible"))
        {
            valid = sink->get<int>("NCE1_Compatible");
        }
        if(source->getOpType() == mv::OpType::Constant && sink->getOpType() == mv::OpType::Conv2D && valid)
        {
            auto weights = sink->getInputTensor(1);
            auto wshape = weights->getShape();

            if(weights->hasAttr("NCE1_WeightTransformed") && weights->get<bool>("NCE1_WeightTransformed") == true)
            {
                ++flowIt;
                continue;
            }

            auto padding = weights->get<std::vector<size_t>>("NCE1_Paddings");

            //for (auto v : padding)
                //std::cout << "PAD: " << v << std::endl;

            //
            auto original_output_channel = wshape[3];
            auto original_output_channel_padding = padding[3];
            auto original_input_channel_padding = padding[2];

            original_output_channel += original_output_channel_padding;

            unsigned floor_output_channel = mv::ceil_division(original_output_channel, 8);

            mv::Shape new_shape = mv::Shape({floor_output_channel, wshape[2], wshape[0], wshape[1], 8});

            std::cout << "oldShape: " << wshape.toString() << std::endl;
            std::cout << "newShape: " << new_shape.toString() << std::endl;


            std::vector<double> new_data(new_shape.totalSize(), 0);
            mv::Tensor backup_tensor("backup", new_shape, weights->getDType(), mv::OrderType::RowMajor, new_data);
            for(unsigned kx = 0; kx < wshape[0]; ++kx)
                for(unsigned ky = 0; ky < wshape[1]; ++ky)
                    for(unsigned ic = 0; ic < wshape[2]; ++ic)
                        for(unsigned oc = 0; oc < wshape[3]; ++oc)
                            backup_tensor.at({oc/8,ic,ky,kx,oc%8}) = weights->at({kx, ky, ic, oc});
            new_data = backup_tensor.getData();

            auto new_op = om.constant(
                new_data,
                new_shape,
                backup_tensor.getDType(),
                backup_tensor.getOrder(),
                source->getName() + "_MxWeights"
            );

            new_op->set<bool>("NCE1_WeightTransformed", true);
            new_op->set<std::vector<size_t>>("NCE1_Paddings",
                {0,
                 original_input_channel_padding,
                 0,
                 0,
                 0
                });

            unsigned i = 0;
            for(; i < sink->inputSlots(); ++i)
                if(sink->getInputTensor(i) == flowIt->getTensor())
                    break;

            auto flowToEliminate = flowIt;
            ++flowIt;

            om.undefineFlow(flowToEliminate);
            sink->erase(std::string("input") + std::to_string(i));
            om.defineFlow(new_op, sink, i);
            om.removeOp(source);

        }
        else
        {
            ++flowIt;
        }
    }
    std::cout << "exiting formatMXweights pass " << std::endl;
}
