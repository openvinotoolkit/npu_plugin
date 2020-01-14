#ifndef MV_OP_HPP_
#define MV_OP_HPP_

#include <string>
#include <array>
#include <map>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    class Op : public ModelElement
    {

        std::vector<Data::TensorIterator> inputs_;
        std::vector<Data::TensorIterator> outputs_;

    public:

        Op(ComputationModel& model, const std::string& opType, const std::string& name,
            const std::vector<Data::TensorIterator>& inputs, const std::vector<std::pair<std::string, Attribute>> & args, bool checkInputSize = true, bool checkArgs = true);

        virtual ~Op();

        std::string getOpType() const;
        bool hasTypeTrait(const std::string& typeTrait) const;

        void setInputTensor(Data::TensorIterator tensor, std::size_t idx, bool cascade = true);
        unsigned addInputTensor(Data::TensorIterator tensor);

        Data::TensorIterator getInputTensor(std::size_t idx);
        Data::TensorIterator getInputTensor(const std::string& label);
        std::vector<Data::TensorIterator> getInputTensor();
        Data::TensorIterator getOutputTensor(std::size_t idx);
        Data::TensorIterator getOutputTensor(const std::string& label);
        std::vector<Data::TensorIterator> getOutputTensor();
        std::map<std::string, mv::Attribute> getAttrs(const std::vector<std::string>& forbiddenKeys = {}) const;

        size_t getOutputSize() const {
          if (outputs_.empty()) { return 0UL;}
          return (*outputs_.front()).getClusterSize();
        }

        std::size_t inputSlots() const;
        std::size_t outputSlots() const;

        std::string getLogID() const override;

    };

}

#endif // COMPUTATION_OP_HPP_
