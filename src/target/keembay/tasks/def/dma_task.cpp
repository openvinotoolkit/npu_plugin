#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/target/keembay/dma_direction.hpp"

namespace mv
{

    namespace op_dma
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
            mv::Tensor toPush(*inputs[0]);
            outputs.push_back(std::move(toPush));
            outputs[0].setName(":0");
            outputs[0].erase("flows");

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

    }

    namespace op {
        MV_REGISTER_OP(DMATask)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::DmaDirection>("direction")
        .setInputCheck(op_dma::inputCheckFcn)
        .setOutputDef(op_dma::outputDefFcn)
        .setTypeTrait({"executable"});
    }

}
