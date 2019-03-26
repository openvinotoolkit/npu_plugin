#ifndef CONTROL_MODEL_HPP_
#define CONTROL_MODEL_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{

    class ControlModel : public ComputationModel
    {

    public:

        ControlModel(ComputationModel &ComputationModel);
        virtual ~ControlModel();

        Control::OpListIterator switchContext(Data::OpListIterator other);

        Control::OpListIterator getFirst();
        Control::OpListIterator getLast();
        Control::OpListIterator opEnd();
        Control::FlowListIterator getInput();
        Control::FlowListIterator getOutput();
        Control::FlowListIterator flowEnd();

        void addGroupElement(Control::OpListIterator element, GroupIterator group);
        void addGroupElement(Control::FlowListIterator element, GroupIterator group);
        void removeGroupElement(Control::OpListIterator element, GroupIterator group);
        void removeGroupElement(Control::FlowListIterator element, GroupIterator group);
        std::vector<Control::OpListIterator> topologicalSort();
        void transitiveReduction();
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        Control::StageIterator addStage();
        Control::StageIterator getStage(std::size_t stageIdx);
        void removeStage(Control::StageIterator stage);
        void addToStage(Control::StageIterator stage, Control::OpListIterator op);
        void addToStage(Control::StageIterator stage, Data::OpListIterator op);
        void removeFromStage(Control::OpListIterator op);
        std::size_t stageSize() const;

        Control::StageIterator stageBegin();
        Control::StageIterator stageEnd();

        std::vector<Control::OpListIterator> getStageMembers(Control::StageIterator stage);

        Control::FlowListIterator defineFlow(Control::OpListIterator sourceOp, Control::OpListIterator sinkOp);
        Control::FlowListIterator defineFlow(Data::OpListIterator sourceOp, Data::OpListIterator sinkOp);
        void undefineFlow(Control::FlowListIterator flow);
        void undefineFlow(Data::FlowListIterator flow);

        virtual std::string getLogID() const override;

    };

}

#endif // CONTROL_MODEL_HPP_
