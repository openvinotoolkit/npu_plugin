#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/tensor/tiling.hpp"

namespace mv
{

    namespace op_hw_convert
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            if (inputs.size() != 1) {
                errMsg = "Invalid number of inputs - must be 1, has " + std::to_string(inputs.size());
                return {false, 0};
            }
            
            auto inputDtype = inputs[0]->getDType();
            auto outputDtype = args.at("dType").get<mv::DType>();
            if (!(inputDtype == mv::DType("UInt8") && outputDtype == mv::DType("Float16")) &&
                !(inputDtype == mv::DType("Float16") && outputDtype == mv::DType("UInt8")))
            {
                errMsg = "Only U8->FP16 and FP16->U8 conversions are supported by HwConvert";
                return {false, 0};
            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto outputDtype = args.at("dType").get<mv::DType>();
            outputs.emplace_back(":0", inputs[0]->getShape(), outputDtype, inputs[0]->getOrder());

            // If this is U8->FP16 set output quant params for dequantization
            if (outputDtype == mv::DType("Float16"))
                outputs[0].setQuantParams(mv::QuantizationParams::initial());
        };
    }

    namespace op
    {
        MV_REGISTER_OP(HwConvert)
            .setInputs({"data"})
            .setOutputs({"output"})
            .setArg<mv::DType>("dType")
            .setInputCheck(op_hw_convert::inputCheckFcn)
            .setOutputDef(op_hw_convert::outputDefFcn)
            .setTypeTrait({"executable", "exposed", "optimizable"});
    }
}
