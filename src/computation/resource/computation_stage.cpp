#include "include/mcm/computation/resource/computation_stage.hpp"

bool mv::ComputationStage::markMembmer_(ComputationElement &member)
{
    if (member.hasAttr("opType"))
    {
        if (member.getAttr("executable").getContent<bool>())
        {
            if (!member.hasAttr("stage"))
            {
                member.addAttr("stage", AttrType::UnsingedType, getAttr("idx").getContent<unsigned_type>());
                return true;
            }
            else
            {
                logger_.log(Logger::MessageType::MessageWarning, "Stage '" + name_ + "' - failed appending member '" + member.getName() + "' that was already assigned to the stage 'stage_" + Printable::toString(member.getAttr("stage").getContent<unsigned_type>()) + "'");
            }
        }
        else
        {
            logger_.log(Logger::MessageType::MessageWarning, "Stage '" + name_ + "' - failed appending member '" + member.getName() + "' of invalid type '" + member.getAttr("opType").getContentStr() + "'");
        }
    }    
    else
    {
        logger_.log(Logger::MessageType::MessageWarning, "Stage '" + name_ + "' - failed appending non-op member '" + member.getName() + "'");
    }
    
    return false;
    
}

bool mv::ComputationStage::unmarkMembmer_(ComputationElement &member)
{

    if (member.hasAttr("stage"))
    {

        member.removeAttr("stage");
        return true;
        
    }

    return false; 

}

mv::ComputationStage::ComputationStage(unsigned_type idx) :
ComputationGroup("stage_" + Printable::toString(idx))
{
    addAttr("idx", AttrType::UnsingedType, idx);
}

mv::string mv::ComputationStage::toString() const
{

    string result = "stage '" + name_ + "'";

    unsigned_type idx = 0;
    for (auto it = members_.begin(); it != members_.end(); ++it)
        result += "\n'member_" + Printable::toString(idx++) + "': " + (*it)->getName();

    return result + ComputationElement::toString();

}

bool mv::ComputationStage::operator <(ComputationElement &other)
{
    return getAttr("idx").getContent<unsigned_type>() < other.getAttr("idx").getContent<unsigned_type>();
}