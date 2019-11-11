#ifndef MV_OP_REGISTRY_HPP_
#define MV_OP_REGISTRY_HPP_

#include <string>
#include <fstream>
#include <cctype>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/computation/op/op_entry.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/master_error.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

namespace mv
{

    namespace op
    {

        class OpRegistry : public Registry<OpRegistry, std::string, OpEntry>
        {

            static const std::string recordedCompModelHeaderPath_;
            static const std::string recordedCompModelSourcePath_;

            /**
             * @brief Legal op types traits
             */
            std::set<std::string> typeTraits_;

            static std::string getCompositionDeclSig_(const std::string& opType, bool args, bool types, bool defaultArgs, bool call, bool recordedModel);
            static std::string getCompositionDecl_(const std::string& opType);
            static std::string getCompositionDef_(const std::string& opType, const std::string& eol = "\n", const std::string& tab = "    ");
            static std::string getCompositionCall_(const std::string& opType);
            static std::string getStringifiedInputsCall_(const std::string opType, const std::string& indent, const std::string eol);
            static std::string getStringifiedOutputsCall_(const std::string opType, const std::string& indent, const std::string eol);
            static std::vector<std::string> getStringifiedArgsCall_(const std::string opType);
            static std::string getLabelNameStringifyCall_(const std::string& label, const std::string& name, std::size_t idx,
                const std::string& indent, const std::string& eol);

        public:

			OpRegistry();
            static OpRegistry& instance();

            static std::vector<std::string> getOpTypes(std::initializer_list<std::string> traits = {});

            static bool checkOpType(const std::string& opType);
            static std::vector<std::string> getArgsList(const std::string& opType);
            static std::vector<std::pair<std::string, Attribute>> getOptionalArgsList(const std::string& opType);
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
            static bool hasTypeTrait(const std::string& opType, const std::string& trait);
            static bool checkExtraInputs(const std::string& opType);


            static void generateCompositionAPI(const std::string& metaDir, const std::string& eol = "\n", const std::string& tab = "    ");
            
        private:
            static void outputMandatoryArgList(std::vector<std::string>& mandatoryArgsList, OpEntry* const opPtr, std::string& mandatoryArgsDef, bool types);
            static void outputOptionalArgList(std::vector<std::pair<std::string, mv::Attribute>>& optionalArgsList, OpEntry* const opPtr, std::string& optionalArgsDef, bool types, bool defaultArgs);
            static void defineOpOutput(std::string& output, const std::string& eol, const std::string& opType, OpEntry* const opPtr, std::string token, bool inputVectorTypes, bool checkInputs, bool copiedOp, const std::string& tab);
        };

        #define MV_REGISTER_OP(Name)															\
            static ATTRIBUTE_UNUSED(OpEntry& CONCATENATE(__ ## OpEntry ## __, __COUNTER__)) =	\
                mv::op::OpRegistry::instance().enter(STRV(Name))


    }

}

#endif // MV_OP_REGISTRY_HPP_
