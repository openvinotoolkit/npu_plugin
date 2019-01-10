#include "include/mcm/target/keembay/tasks/nce2_task_api.hpp"

mv::Data::TensorIterator mv::createDPUTask(mv::BaseOpModel& om, mv::Data::OpListIterator opIt, const std::string& name)
{
    std::vector<std::pair<std::string, mv::Attribute>> list;
    //Attrs vector list must be constructed
    auto attributesKeys = opIt->attrsKeys();
    for(auto attrKey : attributesKeys)
        list.push_back(std::make_pair(attrKey, opIt->get(attrKey)));

    list.push_back(std::make_pair("taskOp", opIt->getOpType()));

    auto inputs = opIt->getInputTensor();
    return om.defineOp("VPUTask", inputs, list, name);
}
