#ifndef STAGE_HPP_
#define STAGE_HPP_

#include "include/mcm/computation/model/computation_group.hpp"
#include "include/mcm/computation/op/op_type.hpp"

namespace mv
{

    class ComputationStage : public ComputationGroup
    {

    protected:

        virtual bool markMembmer_(Element &member) override;
        virtual bool unmarkMembmer_(Element &member) override;

    public:

        ComputationStage(std::size_t idx);
        std::size_t getIdx() const;
        std::string toString() const override;
        bool operator <(ComputationStage &other);
        virtual std::string getLogID() const override;

    };

}

#endif // STAGE_HPP_