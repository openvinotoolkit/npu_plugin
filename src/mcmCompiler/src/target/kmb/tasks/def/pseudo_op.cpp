#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
  namespace op_pseudo_op {

    static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
        const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
        [](const std::vector<Data::TensorIterator>& inputs,
            const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> 
        std::pair<bool, std::size_t> { return {true, 0}; };

        static std::function<void(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs,
               const std::map<std::string, Attribute>& args,
               std::vector<Tensor>& outputs) {

              std::vector<std::size_t> inputShape0(inputs[0]->getShape());
              auto dTypeToUse = inputs[0]->getDType();
             
              outputs.push_back( mv::Tensor(":0", mv::Shape(inputShape0),
                    dTypeToUse, inputs[0]->getOrder()) ); 
        };

  } // namespace op_pseudo_op //

  namespace op {
      MV_REGISTER_OP(PseudoOp)
      .setInputs({"inputs"})
      .setOutputs({"output"})
      .setVariableInputNum(true)
      .setInputCheck(op_pseudo_op::inputCheckFcn)
      .setOutputDef(op_pseudo_op::outputDefFcn);
  }

} // namespace mv //

