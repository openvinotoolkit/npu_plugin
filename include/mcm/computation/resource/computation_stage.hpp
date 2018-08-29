#ifndef STAGE_HPP_
#define STAGE_HPP_

#include "include/mcm/computation/model/computation_group.hpp"

namespace mv
{

    class ComputationStage : public ComputationGroup
    {

        std::size_t idx_;

    protected:

        virtual bool markMembmer_(ComputationElement &member);
        virtual bool unmarkMembmer_(ComputationElement &member);
        virtual std::string getLogID_() const override;

    public:

        ComputationStage(std::size_t idx);
        std::size_t getIdx() const;
        std::string toString() const;
        bool operator <(ComputationStage &other);
        
    };

}

#endif // STAGE_HPP_