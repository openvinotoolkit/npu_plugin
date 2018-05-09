#ifndef COMPUTATION_MODEL_HPP_
#define COMPUTATION_MODEL_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/api/compositional_model.hpp"
#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/model_constant.hpp"
#include "include/fathom/computation/tensor/model_variable.hpp"
#include "include/fathom/computation/op/input.hpp"
#include "include/fathom/computation/op/output.hpp"
#include "include/fathom/computation/op/conv.hpp"
#include "include/fathom/computation/logger/stdout.hpp"

namespace mv
{

    class ComputationModel : public CompositionalModel
    {
    
        static allocator allocator_;
        computation_graph ops_graph;
        Logger &logger_;
        OpListIterator input_;
        OpListIterator output_;

        static Logger &getLogger(Logger::VerboseLevel verboseLevel, bool logTime);

    public:

        ComputationModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        ComputationModel(Logger &logger);
        const OpListIterator input(const Shape &shape, DType dType, Order order, const string &name = "");
        const OpListIterator output(OpListIterator &predecessor, const string &name = "");
        OpListIterator convolutional(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, const string &name = "");
        bool addAttr(OpListIterator &op, const string &name, const Attribute &attr);
        const Logger& logger() const;

    };

}

#endif // COMPUTATION_MODEL_HPP_