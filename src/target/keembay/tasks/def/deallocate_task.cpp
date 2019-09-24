#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{

    namespace op_deallocate
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
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::vector<Tensor>& )
        {

        };
    
    }

    namespace op {
        MV_REGISTER_OP(Deallocate)
        .setInputs({"inputs"})
        .setInputCheck(op_deallocate::inputCheckFcn)
        .setOutputDef(op_deallocate::outputDefFcn)
        .setTypeTrait({"executable"});
    }
}
