#ifndef MV_OP_ENTRY_HPP_
#define MV_OP_ENTRY_HPP_

#include <string>
#include <functional>
#include <set>
#include <map>
#include <typeindex>
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/base/exception/master_error.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

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
            std::map<std::string, std::type_index> args_;

        public:

            OpEntry(const std::string& opType) :
            opType_(opType)
            {

            }

            inline OpEntry& setInputs(std::vector<std::string> labels)
            {

                inputLabels_ = labels;
                return *this;

            }

            inline OpEntry& setOutputs(std::vector<std::string> labels)
            {

                outputLabels_ = labels;
                return *this;

            }

            inline OpEntry& setInputCheck(const std::function<std::pair<bool, std::size_t>
                (const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::string&)>& inputCheck)
            {
                inputCheck_ = inputCheck;
                return *this;
            }

            inline OpEntry& setOutputDef(std::function<void(const std::vector<Data::TensorIterator>&,
                const std::map<std::string, Attribute>&, std::vector<Tensor>&)>& outputDef)
            {
                outputDef_ = outputDef;
                return *this;
            }

            inline OpEntry& setDescription(const std::string& description)
            {
                description_ = description;
                return *this;
            }

            template <class AttrType>
            inline OpEntry& setArg(const std::string& name)
            {

                if (!attr::AttributeRegistry::checkType<AttrType>())
                    throw AttributeError("OpEntry", "Attempt of setting argument of an unregistered attribute type "
                        + std::string(typeid(AttrType).name()) + " \"" + name + "\" for ");

                args_.emplace(name, typeid(AttrType));
                return *this;

            }

            inline const std::string getDescription() const
            {
                return description_;
            }

            inline std::size_t getInputsCount() const
            {
                return inputLabels_.size();
            }

            inline std::size_t getOutputsCount() const
            {
                return outputLabels_.size();
            }

            inline bool hasArg(const std::string& name) const
            {
                return args_.find(name) != args_.end();
            }

            inline std::type_index argType(const std::string& name) const
            {
                if (!hasArg(name))
                    throw OpError(*this, "Attempt of checking the type of an non-existing argument \"" + name + "\"");
                return args_.at(name);
            }

            inline std::vector<std::string> argsList() const
            {
                std::vector<std::string> list;
                list.reserve((args_.size()));
                for (auto &arg : args_)
                    list.push_back(arg.first);
                return list;
            }

            inline std::pair<bool, std::size_t> checkInputs(const std::vector<Data::TensorIterator>& inputs, 
                const std::map<std::string, Attribute>& args, std::string& errMsg)
            {
                return inputCheck_(inputs, args, errMsg);
            }

            inline void getOutputsDef(const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                std::vector<Tensor>& outputs)
            {
                outputDef_(inputs, args, outputs);
            }

            inline std::string getInputLabel(std::size_t idx)
            {

                if (idx >= inputLabels_.size())
                    throw IndexError(*this, idx, "Passed input index exceeds inputs count registered for the op type " + opType_);

                return inputLabels_[idx];
            
            }

            inline const std::vector<std::string>& getInputLabel()
            {
                return inputLabels_;
            }

            inline std::string getOutputLabel(std::size_t idx)
            {

                if (idx >= outputLabels_.size())
                    throw IndexError(*this, idx, "Passed input index exceeds outputs count registered for the op type " + opType_);

                return outputLabels_[idx];
            
            }

            inline const std::vector<std::string>& getOutputLabel()
            {
                return outputLabels_;
            }

            std::string getLogID() const override
            {
                return "OpEntry:" + opType_;
            }


        };

    }

}

#endif // MV_OP_ENTRY_HPP_