#ifndef MV_OP_ENTRY_HPP_
#define MV_OP_ENTRY_HPP_

#include <string>
#include <functional>
#include <set>
#include <map>
#include <array>
#include <vector>
#include <typeindex>
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/base/exception/master_error.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"

namespace mv
{

    namespace op
    {

        class OpEntry : public LogSender
        {
            
            std::string opType_;
            std::string description_;
            std::set<std::string> opTraits_;
            std::function<std::pair<bool, std::size_t>
                (const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::string&)> inputCheck_;
            std::vector<std::string> inputLabels_;
            std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
                std::vector<Tensor>&)> outputDef_;
            std::vector<std::string> outputLabels_;
            std::vector<std::tuple<std::string, std::type_index, Attribute>> args_;
            std::set<std::string> typeTraits_;

        public:

            OpEntry(const std::string& opType);

            OpEntry& setInputs(std::vector<std::string> labels);
            OpEntry& setOutputs(std::vector<std::string> labels);
            OpEntry& setInputCheck(const std::function<std::pair<bool, std::size_t>
                (const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::string&)>& inputCheck);
            OpEntry& setOutputDef(std::function<void(const std::vector<Data::TensorIterator>&,
                const std::map<std::string, Attribute>&, std::vector<Tensor>&)>& outputDef);
            OpEntry& setDescription(const std::string& description);
            OpEntry& setTypeTrait(const std::string& trait);
            OpEntry& setTypeTrait(std::initializer_list<std::string> traits);

            const std::string getDescription() const;
            std::size_t getInputsCount() const;
            std::size_t getOutputsCount() const;
            bool hasArg(const std::string& name) const;
            std::type_index argType(const std::string& name) const;
            std::vector<std::string> argsList() const;
            std::pair<bool, std::size_t> checkInputs(const std::vector<Data::TensorIterator>& inputs, 
                const std::map<std::string, Attribute>& args, std::string& errMsg);
            void getOutputsDef(const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                std::vector<Tensor>& outputs);
            std::string getInputLabel(std::size_t idx);
            const std::vector<std::string>& getInputLabel();
            std::string getOutputLabel(std::size_t idx);
            const std::vector<std::string>& getOutputLabel();
            bool hasTypeTrait(const std::string& trait);
            const std::set<std::string>& getTypeTraits();

            std::string getLogID() const override;

            template <class AttrType>
            inline OpEntry& setArg(const std::string& name, Attribute val = Attribute())
            {

                if (!attr::AttributeRegistry::checkType<AttrType>())
                    throw AttributeError("OpEntry", "Attempt of setting argument of an unregistered attribute type "
                        + std::string(typeid(AttrType).name()) + " \"" + name + "\" for ");

                args_.push_back({name, typeid(AttrType), val});
                return *this;

            }


        };

    }

}

#endif // MV_OP_ENTRY_HPP_