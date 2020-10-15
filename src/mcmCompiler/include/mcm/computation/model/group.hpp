#ifndef GROUP_HPP_
#define GROUP_HPP_

#include <algorithm>
#include <vector>
#include <string>
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/base/exception/logic_error.hpp"

namespace mv
{

    class Group;
    using GroupIterator = IteratorDetail::ModelValueIterator<std::map<std::string, std::shared_ptr<Group>>::iterator, Group>;
    class ComputationModel;

    class Group : public ModelElement
    {
        
        template <class IteratorType>
        void include_(IteratorType it, std::vector<std::string>& memberList)
        {   
            if (std::find(memberList.begin(), memberList.end(), it->getName()) == memberList.end())
            {
                memberList.push_back(it->getName());
                if (!it->hasAttr("groups"))
                    it->template set<std::vector<std::string>>("groups", { getName() });
                else
                    it->template get<std::vector<std::string>>("groups").push_back(getName());
            }
        }

        template <class IteratorType>
        void exclude_(IteratorType it, std::vector<std::string>& memberList)
        {   
            if (std::find(memberList.begin(), memberList.end(), it->getName()) != memberList.end())
            {
                memberList.erase(std::remove(memberList.begin(), memberList.end(), it->getName()), memberList.end());
                if (it->hasAttr("groups"))
                {
                    std::vector<std::string>& groupsList = it->template get<std::vector<std::string>>("groups");
                    groupsList.erase(std::remove(groupsList.begin(), groupsList.end(), getName()), groupsList.end());
                }
                else
                    throw LogicError(*this, "Marked member " + it->getName() + " was not labelled");
            }
        }

        template <class IteratorType>
        bool isMember_(IteratorType it, const std::vector<std::string>& memberList) const
        {   
            return std::find(memberList.begin(), memberList.end(), it->getName()) != memberList.end();
        }

    public:

        Group(ComputationModel &model, const std::string &name);

        void include(Data::OpListIterator op);
        void include(Control::OpListIterator op);
        void include(Data::FlowListIterator flow);
        void include(Control::FlowListIterator flow);
        void include(Data::TensorIterator tensor);
        void include(GroupIterator group);
        //void include(Control::StageIterator stage);

        void exclude(Data::OpListIterator op);
        void exclude(Control::OpListIterator op);
        void exclude(Data::FlowListIterator flow);
        void exclude(Control::FlowListIterator flow);
        void exclude(Data::TensorIterator tensor);
        void exclude(GroupIterator group);
        //void exclude(Control::StageIterator stage);

        bool isMember(Data::OpListIterator op) const;
        bool isMember(Control::OpListIterator op) const;
        bool isMember(Data::FlowListIterator flow) const;
        bool isMember(Control::FlowListIterator flow) const;
        bool isMember(Data::TensorIterator tensor) const;
        bool isMember(GroupIterator group) const;
        //bool isMember(Control::StageIterator stage) const;

        std::vector<Data::OpListIterator> getOpMembers();
        std::vector<Data::FlowListIterator> getDataFlowMembers();
        std::vector<Control::FlowListIterator> getControlFlowMembers();
        std::vector<Data::TensorIterator> getTensorMembers();
        std::vector<GroupIterator> getGroupMembers();

        void clear();
        std::size_t size() const;
        std::string toString() const override;
        std::string getLogID() const override;

    };

}

#endif // GROUP_HPP_