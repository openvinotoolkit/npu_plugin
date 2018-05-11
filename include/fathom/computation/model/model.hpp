#ifndef COMPUTATION_MODEL_HPP_
#define COMPUTATION_MODEL_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/api/compositional_model.hpp"
#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator/op_iterator.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/populated.hpp"
#include "include/fathom/computation/tensor/unpopulated.hpp"
#include "include/fathom/computation/logger/stdout.hpp"
#include "include/fathom/computation/flow/data.hpp"

namespace mv
{

    class ComputationModel
    {
    
    protected:

        struct TensorOrderComparator
        {
            bool operator()(const allocator::owner_ptr<ModelTensor> &lhs, const allocator::owner_ptr<ModelTensor> &rhs)
            {
                return lhs->getID() < rhs->getID();
            }
        };

        static allocator allocator_;

        /*
        There are two reasons to store all member variables that are non-static members as either references or smart pointers provided by
        the Allocator concept
            - for objects that are containers - enforcing to be failure safe by using Allocator's factory methods (no possibility of 
            having unhandled bad allocation errors, particularly STL exceptions)
            - obtaining a capability of shallow coping the ComputationModel that is exploited by e.g. switchable contexts (OpModel, DataModel)
        */
        allocator::owner_ptr<computation_graph> ops_graph_;
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<UnpopulatedTensor>, TensorOrderComparator>> flowTensors_;
        allocator::owner_ptr<allocator::set<allocator::owner_ptr<PopulatedTensor>, TensorOrderComparator>> parameterTensors_;
        const allocator::owner_ptr<Logger> defaultLogger_;
        Logger &logger_;
        OpListIterator input_;
        OpListIterator output_;

    public:

        ComputationModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        ComputationModel(Logger &logger);

        /**
         * @brief Copy constructor performing shallow copy
         * 
         * @param other Object that will share all members with the new one
         */
        ComputationModel(const ComputationModel &other);

        virtual ~ComputationModel() = 0;
        bool isValid() const;
        const Logger& logger() const;

    };

}

#endif // COMPUTATION_MODEL_HPP_