#ifndef COMPUTATION_MODEL_HPP_
#define COMPUTATION_MODEL_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/api/compositional_model.hpp"
#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/model_populated.hpp"
#include "include/fathom/computation/tensor/model_unpopulated.hpp"
#include "include/fathom/computation/op/input.hpp"
#include "include/fathom/computation/op/output.hpp"
#include "include/fathom/computation/op/conv.hpp"
#include "include/fathom/computation/logger/stdout.hpp"
#include "include/fathom/computation/flow/data.hpp"

namespace mv
{

    class ComputationModel : public CompositionalModel
    {
    
        struct TensorOrderComparator
        {
            bool operator()(const allocator::owner_ptr<ModelTensor> &lhs, const allocator::owner_ptr<ModelTensor> &rhs)
            {
                return lhs->getID() < rhs->getID();
            }
        };

        static allocator allocator_;
        computation_graph ops_graph_;
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<UnpopulatedModelTensor>, TensorOrderComparator>> flowTensors_;
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<PopulatedModelTensor>, TensorOrderComparator>> parameterTensors_;
        const allocator::owner_ptr<Logger> defaultLogger_;
        Logger &logger_;
        OpListIterator input_;
        OpListIterator output_;

    public:

        ComputationModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        ComputationModel(Logger &logger);
        const OpListIterator input(const Shape &shape, DType dType, Order order, const string &name = "");
        const OpListIterator output(OpListIterator &predecessor, const string &name = "");
        OpListIterator conv2D(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        bool addAttr(OpListIterator &op, const string &name, const Attribute &attr);
        bool isValid() const;
        const Logger& logger() const;

    };

}

#endif // COMPUTATION_MODEL_HPP_