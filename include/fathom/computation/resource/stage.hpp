#ifndef STAGE_HPP_
#define STAGE_HPP_

#include "include/fathom/computation/model/group.hpp"

namespace mv
{

    class ComputationStage : public ComputationGroup
    {

        unsigned_type idx_;

    protected:

        virtual bool markMembmer(ComputationElement &member)
        {
            if (!member.hasAttr("stage"))
            {
                member.addAttr("stage", AttrType::UnsingedType, idx_);
                return true;
            }
            
            return false;
            
        }

        virtual bool unmarkMembmer(ComputationElement &member)
        {

            if (member.hasAttr("stage"))
            {

                member.removeAttr("stage");
                return true;
                
            }

            return false; 

        }

    public:

        ComputationStage(const Logger &logger, unsigned_type idx) :
        ComputationGroup(logger, "stage_" + Printable::toString(idx)),
        idx_(idx)
        {

        }

    };

}

#endif // STAGE_HPP_