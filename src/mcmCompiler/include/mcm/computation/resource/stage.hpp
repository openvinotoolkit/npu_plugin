#ifndef MV_STAGE_HPP_
#define MV_STAGE_HPP_

#include <algorithm>
#include <vector>
#include <string>
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/base/exception/logic_error.hpp"

namespace mv
{

    class Stage : public ModelElement
    {

    public:

        Stage(ComputationModel& model, std::size_t idx);

        void include(Control::OpListIterator op);
        void exclude(Control::OpListIterator op);

        bool isMember(Control::OpListIterator op) const;
        std::vector<Control::OpListIterator> getMembers();
        void clear();
    
        std::size_t getIdx() const;
        std::string toString() const override;
        std::string getLogID() const override;

        bool operator <(Stage &other);

    };

}

#endif // MV_STAGE_HPP_