#include "include/mcm/computation/resource/computation_stage.hpp"

bool mv::ComputationStage::markMembmer_(Element &member)
{
    if (member.hasAttr("opType"))
    {
        if (member.get<bool>("executable"))
        {
            if (!member.hasAttr("stage"))
            {
                member.set<std::size_t>("stage", getIdx());
                return true;
            }
            else
            {
                log(Logger::MessageType::Warning, "Stage '" + name_ + "' - failed appending member '" + member.getName() + 
                    "' that was already assigned to this stage");
            }
        }
        else
        {
            log(Logger::MessageType::Warning, "Stage '" + name_ + "' - failed appending member '" + member.getName() + 
                "' of invalid type '" + member.get<OpType>("opType").toString() + "'");
        }
    }    
    else
    {
        log(Logger::MessageType::Warning, "Stage '" + name_ + "' - failed appending non-op member '" + 
            member.getName() + "'");
    }
    
    return false;
    
}

bool mv::ComputationStage::unmarkMembmer_(Element &member)
{

    if (member.hasAttr("stage"))
    {

        member.erase("stage");
        return true;
        
    }

    return false; 

}

mv::ComputationStage::ComputationStage(std::size_t idx) :
ComputationGroup("stage_" + std::to_string(idx))
{
    set<std::size_t>("idx", idx);
}

std::size_t mv::ComputationStage::getIdx() const
{
    return get<std::size_t>("idx");
}

std::string mv::ComputationStage::toString() const
{

    std::string result = "stage '" + name_ + "'";

    std::size_t idx = 0;
    for (auto it = members_.begin(); it != members_.end(); ++it)
        result += "\n'member_" + std::to_string(idx++) + "': " + it->lock()->getName();

    return result + Element::attrsToString_();

}

bool mv::ComputationStage::operator <(ComputationStage &other)
{
    return getIdx() < other.getIdx();
}

std::string mv::ComputationStage::getLogID() const
{
    return "Stage:" + std::to_string(getIdx());
}