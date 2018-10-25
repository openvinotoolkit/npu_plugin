#ifndef MV_OP_REGISTRY_HPP_
#define MV_OP_REGISTRY_HPP_

#include <string>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/computation/op/op_entry.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/master_error.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    namespace op
    {

        class OpRegistry : public Registry<std::string, OpEntry>
        {

            /**
             * @brief Legal op types traits
             */
            static const std::set<std::string> typeTraits_;

        public:

            static OpRegistry& instance();

            static bool checkOpType(const std::string& opType);
            static std::vector<std::string> argsList(const std::string& opType);
            static std::type_index argType(const std::string& opType, const std::string& argName);
            static bool checkArgType(const std::string& opType, const std::string& argName, const std::type_index& typeID);
            static std::size_t getInputsCount(const std::string& opType);
            static std::size_t getOutputsCount(const std::string& opType);
            static std::pair<bool, std::size_t> checkInputs(const std::string& opType,
                const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::string& errMsg);
            static void getOutputsDef(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
                const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs);
            static std::string getInputLabel(const std::string& opType, std::size_t idx);
            static const std::vector<std::string>& getInputLabel(const std::string& opType);
            static std::string getOutputLabel(const std::string& opType, std::size_t idx);
            static const std::vector<std::string>& getOutputLabel(const std::string& opType);
            static bool checkTypeTrait(const std::string& typeTrait);
            static const std::set<std::string>& getTypeTraits(const std::string& opType);
            
        };

        #define MV_REGISTER_OP(Name)                          \
            MV_REGISTER_ENTRY(std::string, OpEntry, #Name)    \


    }

}

#endif // MV_OP_REGISTRY_HPP_
