#ifndef COMPUTATION_MODEL_HPP_
#define COMPUTATION_MODEL_HPP_

#include <memory>
#include <map>
#include <string>
#include "include/mcm/graph/conjoined_graph.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/model/iterator/stage.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/flow/data_flow.hpp"
#include "include/mcm/computation/flow/control_flow.hpp"
#include "include/mcm/computation/model/group.hpp"
#include "include/mcm/computation/resource/stage.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/model/runtime_binary.hpp"
#include "include/mcm/target/kmb/dma_direction.hpp"
#include "include/mcm/target/kmb/ppe_layer_type.hpp"
#include "include/mcm/target/kmb/barrier_definition.hpp"
#include "include/mcm/target/kmb/barrier_deps.hpp"

namespace mv
{

    class ComputationModel : public LogSender
    {

    protected:

        std::string name_;
        /*
        There are two reasons to store all member variables that are non-static members as either references or smart pointers provided by
        the Allocator concept
            - for objects that are containers - enforcing to be failure safe by using Allocator's factory methods (no possibility of
            having unhandled bad allocation errors, particularly STL exceptions)
            - obtaining a capability of shallow coping the ComputationModel that is exploited by e.g. switchable contexts (OpModel, DataModel)
        */
        std::shared_ptr<conjoined_graph<Op, DataFlow, ControlFlow>> opsGraph_;
        std::shared_ptr<mv::RuntimeBinary> binary_;
        dataGraph &dataGraph_;
        controlGraph &controlGraph_;
        std::shared_ptr<mv::Element> globalConfigParams_;
        /*
        There are two reasons to use sets as underlying containers:
        - All operations complexity n * log(n)
        - Iterator of set is invalidated only on deletion of pointed element (on the other hand, vector's iterator is invalidated on the resize of the vector)
            - ModelLinearIterators are wrapping containers iterators
        */
        std::shared_ptr<std::unordered_map<std::string, Data::OpListIterator>> ops_;
        std::shared_ptr<std::unordered_map<std::string, Data::FlowListIterator>> dataFlows_;
        std::shared_ptr<std::unordered_map<std::string, Control::FlowListIterator>> controlFlows_;
        std::shared_ptr<std::map<std::string, std::shared_ptr<Tensor>>> tensors_;
        std::shared_ptr<std::map<std::string, std::shared_ptr<Group>>> groups_;
        std::shared_ptr<std::map<std::size_t, std::shared_ptr<Stage>>> stages_;
        std::shared_ptr<std::map<std::string, std::shared_ptr<MemoryAllocator>>> memoryAllocators_;
        std::shared_ptr<std::map<std::string, std::size_t>> opsInstanceCounter_;
        std::shared_ptr<std::map<std::string, std::size_t>> opsIndexCounter_;
        std::shared_ptr<Data::OpListIterator> dataOpEnd_;
        std::shared_ptr<Data::FlowListIterator> dataFlowEnd_;
        std::shared_ptr<Control::OpListIterator> controlOpEnd_;
        std::shared_ptr<Control::FlowListIterator> controlFlowEnd_;
        std::shared_ptr<Data::OpListIterator> input_;
        std::shared_ptr<Data::OpListIterator> output_;

        std::reference_wrapper<ComputationModel> selfRef_;

        Data::TensorIterator findTensor_(const std::string &name);
        void incrementOpsInstanceCounter_(const std::string& opType);
        void decrementOpsInstanceCounter_(const std::string& opType);
        void incrementOpsIndexCounter_(const std::string& opType);

    public:

        ComputationModel(const std::string& name);
        //ComputationModel(mv::json::Value& model);

        /**
         * @brief Copy constructor performing shallow copy
         *
         * @param other Object that will share all members with the new one
         */
        ComputationModel(ComputationModel &other);

        virtual ~ComputationModel() = 0;

        /**
         * @brief Check basic logical cohesion of the computation model. Does not guarantee that the model can executed successfully on the
         * target platform
         *
         * @return true Computation model is valid.
         * @return false Computation model is invalid.
         */
        bool isValid() const;
        bool isValid(Data::TensorIterator it) const;
        bool isValid(Data::OpListIterator it) const;
        bool isValid(Control::OpListIterator it) const;
        bool isValid(Data::FlowListIterator it) const;
        bool isValid(Control::FlowListIterator it) const;
        bool isValid(GroupIterator it) const;
        bool isValid(Control::StageIterator it) const;

        GroupIterator addGroup(const std::string &name);
        GroupIterator groupBegin();
        GroupIterator groupEnd();
        bool hasGroup(const std::string &name);
        GroupIterator getGroup(const std::string& name);

        void addGroupElement(GroupIterator element, GroupIterator group);
        void removeGroupElement(GroupIterator element, GroupIterator group);

        Data::TensorIterator tensorBegin() const;
        Data::TensorIterator tensorEnd() const;
        Data::TensorIterator getTensor(const std::string& name);

        bool checkOp(const std::string& name);
        Data::OpListIterator getOp(const std::string& name);
        std::vector<Data::OpListIterator> getOps(const std::string& opType);
        std::vector<mv::Data::OpListIterator> getOps();
        std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> getOpsOfTypes(const std::vector<std::string> &opTypes);
        Data::FlowListIterator getDataFlow(const std::string& name);
        Control::FlowListIterator getControlFlow(const std::string& name);

        void clear();
        std::shared_ptr<mv::RuntimeBinary> allocateBinaryBuffer(std::string newName, std::size_t newSize);
        std::shared_ptr<mv::RuntimeBinary> allocateBinaryBuffer(std::size_t newSize);
        std::shared_ptr<mv::RuntimeBinary> getBinaryBuffer();

        std::reference_wrapper<ComputationModel> getRef();

        /**
         * @brief Gets global params stored under GlobalConfigParams in Compilation Descriptor
         *
         * @return compilation descriptor Element*/
        std::shared_ptr<mv::Element>getGlobalConfigParams() const;
        void setGlobalConfigParams(mv::Element& element);

        std::string getName() const;
        virtual std::string getLogID() const override;
        json::Value toJSON() const;
        json::Array dataFlowToJSON() const;
        json::Array controlFlowToJSON() const;
        json::Array opsToJSON() const;
        json::Object opsIndexCounterToJSON() const;
        json::Object opsInstanceCounterToJSON() const;
        json::Array stagesToJSON() const;
        json::Array groupsToJSON() const;
        json::Array tensorsToJSON() const;
        bool hasPopulatedTensorsToJSON() const;
        json::Object memoryAllocatorsToJSON() const;
        json::Object sourceOpsToJSON() const;
    };

}

#endif // COMPUTATION_MODEL_HPP_
