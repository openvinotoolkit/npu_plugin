#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/target/keembay/dma_direction.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&args, std::vector<Tensor>& outputs)
        {
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

            if (inputs[0]->isPopulated())
                outputs[0].populate(inputs[0]->getData());
            if (inputs[0]->isSparse())
            {
                //Sparsity map shallow copy
                outputs[0].setSparse(inputs[0]->getSparsityMap(), inputs[0]->getStorageElement());
            }
            if (inputs[0]->hasAttr("channelLength"))
                outputs[0].set<int>("channelLength", inputs[0]->get<int>("channelLength"));
            if (args.at("direction").get<mv::DmaDirection>() == mv::DmaDirectionEnum::DDR2CMX)
            {
                mv::Tensor::MemoryLocation outputLocation("CMX");
                outputs[0].set<mv::Tensor::MemoryLocation>("Location", outputLocation);
            }
            else if (args.at("direction").get<mv::DmaDirection>() == mv::DmaDirectionEnum::CMX2DDR)
            {
                mv::Tensor::MemoryLocation outputLocation("DDR");
                outputs[0].set<mv::Tensor::MemoryLocation>("Location", outputLocation);
            }
        };

        MV_REGISTER_OP(DMATask)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::DmaDirection>("direction")
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{},{},{}))
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable"});

    }

}
