#include "include/mcm/target/keembay/tasks/nce2_task_api.hpp"

mv::Data::TensorIterator mv::createDPUTask(mv::BaseOpModel& om, const std::vector<mv::Data::TensorIterator>& inputs, const std::string& opType, std::vector<std::pair<std::string, mv::Attribute>>& list, const std::string& name)
{
    list.push_back(std::make_pair("taskOp", opType));
    return om.defineOp("DPUTask", inputs, list, name, true);
}


mv::Data::TensorIterator mv::createDPUTask(mv::BaseOpModel& om, mv::Data::OpListIterator opIt, const std::string& name)
{
    std::vector<std::pair<std::string, mv::Attribute>> list;
    //Attrs vector list must be constructed
    auto attributesKeys = opIt->attrsKeys();
    for(auto attrKey : attributesKeys)
        if(attrKey != "opType")
            list.push_back(std::make_pair(attrKey, opIt->get(attrKey)));

    return createDPUTask(om, opIt->getInputTensor(), opIt->getOpType(), list, name);
}


mv::Data::TensorIterator mv::createDMATask(mv::BaseOpModel& om, mv::Data::TensorIterator data0, mv::DmaDirection direction, const std::string& name)
{
    return om.defineOp("DMATask", {data0}, {{"direction", direction}}, name);
}
