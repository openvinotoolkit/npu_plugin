#include "include/mcm/computation/op/op_registry.hpp"

#include <sstream>

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            (void) inputs; // unused

            double min = args.at("min").get<double>();
            double max = args.at("max").get<double>();

            if (min < 0 || min > max)
            {
                std::stringstream err;
                err << "Wrong min, max parameters: min=" << min << ", max=" << max;
                errMsg = err.str();
                return {false, 1};
            }

            return {true, 0};
        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {

            outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));

        };
    
        MV_REGISTER_OP(Clamp)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<double>("min")
        .setArg<double>("max")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
