#ifndef STAGE_HPP_
#define STAGE_HPP_

#include "include/mcm/computation/model/computation_group.hpp"

namespace mv
{

    class ComputationStage : public ComputationGroup
    {

    protected:

        virtual bool markMembmer_(ComputationElement &member);
        virtual bool unmarkMembmer_(ComputationElement &member);
    public:

        ComputationStage(std::size_t idx);
        std::string toString() const;
        bool operator <(ComputationElement &other);
        
    };

}

#endif // STAGE_HPP_