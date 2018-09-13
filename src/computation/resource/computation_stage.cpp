#include "include/mcm/computation/resource/computation_stage.hpp"

/*bool mv::ComputationStage::markMembmer_(ComputationElement &member)
{
    if (member.hasAttr("opType"))
    {
        if (member.getAttr("executable").getContent<bool>())
        {
            if (!member.hasAttr("stage"))
            {
                member.addAttr("stage", AttrType::UnsignedType, idx_);
                return true;
            }
            else
            {
                log(Logger::MessageType::MessageWarning, "Stage '" + name_ + "' - failed appending member '" + member.getName() + "' that was already assigned to the stage 'stage_" + Printable::toString(member.getAttr("stage").getContent<std::size_t>()) + "'");
            }
        }
        else
        {
            log(Logger::MessageType::MessageWarning, "Stage '" + name_ + "' - failed appending member '" + member.getName() + "' of invalid type '" + member.getAttr("opType").getContentStr() + "'");
        }
    }    
    else
    {
        log(Logger::MessageType::MessageWarning, "Stage '" + name_ + "' - failed appending non-op member '" + member.getName() + "'");
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

mv::ComputationStage::ComputationStage(std::size_t idx) :
ComputationGroup("stage_" + Printable::toString(idx)),
idx_(idx)
{

}

std::size_t mv::ComputationStage::getIdx() const
{
    return idx_;
}

std::string mv::ComputationStage::toString() const
{

    std::string result = "stage '" + name_ + "'";

    std::size_t idx = 0;
    for (auto it = members_.begin(); it != members_.end(); ++it)
        result += "\n'member_" + Printable::toString(idx++) + "': " + it->lock()->getName();

    return result + ComputationElement::toString();

}

bool mv::ComputationStage::operator <(ComputationStage &other)
{
    return idx_ < other.idx_;
}

std::string mv::ComputationStage::getLogID_() const
{
    return "Stage " + Printable::toString(idx_);
}
*/