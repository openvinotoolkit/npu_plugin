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

            static bool checkOpType(const std::string& opType)
            {
                return instance().find(opType) != nullptr;
            }

            static std::vector<std::string> argsList(const std::string& opType)
            {
                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of obtaining the arguments list for an unregistered op type " + opType);
                
                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType + " not found in the op registry");

                return opPtr->argsList();
                
            }

            static std::type_index argType(const std::string& opType, const std::string& argName)
            {

                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of checking the arguments type for an unregistered op type " + opType);
                
                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType + " not found in the op registry");

                return opPtr->argType(argName);

            }

            static bool checkArgType(const std::string& opType, const std::string& argName, const std::type_index& typeID)
            {
                return typeID == argType(opType, argName);
            }

            static std::size_t getInputsCount(const std::string& opType)
            {
                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of checking inputs count for an unregistered op type " + opType);
                
                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->getInputsCount();

            }

            static std::size_t getOutputsCount(const std::string& opType)
            {
                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of checking outputs count for an unregistered op type " + opType);
                
                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->getOutputsCount();

            }

            static std::pair<bool, std::size_t> checkInputs(const std::string& opType,
                const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::string& errMsg)
            {
                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of executing an inputs check for an unregistered op type " + opType);
                
                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->checkInputs(inputs, args, errMsg);
            }

            static void getOutputsDef(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
                const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
            {

                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of executing an inputs check for an unregistered op type " + opType);
                
                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                opPtr->getOutputsDef(inputs, args, outputs);

            }

            static std::string getInputLabel(const std::string& opType, std::size_t idx)
            {

                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of obtaining an input label for an unregistered op type " + opType);

                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->getInputLabel(idx);

            }

            static const std::vector<std::string>& getInputLabel(const std::string& opType)
            {
                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of obtaining an input labels for an unregistered op type " + opType);

                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->getInputLabel();
            }

            static std::string getOutputLabel(const std::string& opType, std::size_t idx)
            {

                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of obtaining an output label for an unregistered op type " + opType);

                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->getOutputLabel(idx);

            }

            static const std::vector<std::string>& getOutputLabel(const std::string& opType)
            {

                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of obtaining an output labels for an unregistered op type " + opType);

                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->getOutputLabel();

            }

            static bool checkTypeTrait(const std::string& typeTrait)
            {
                if (typeTraits_.find(typeTrait) != typeTraits_.end())
                    return true;
                return false;
            }

            static const std::set<std::string>& getTypeTraits(const std::string& opType)
            {
                if (!checkOpType(opType))
                    throw OpError("OpRegistry", "Attempt of obtaining type traits for an unregistered op type " + opType);

                OpEntry* const opPtr = instance().find(opType);

                if (!opPtr)
                    throw MasterError("OpRegistry", "Registered op type " + opType +
                        " not found in the op registry");

                return opPtr->getTypeTraits();
            }

        };

        #define MV_REGISTER_OP(Name)                          \
            MV_REGISTER_ENTRY(std::string, OpEntry, #Name)    \


    }

}

#endif // MV_OP_REGISTRY_HPP_
