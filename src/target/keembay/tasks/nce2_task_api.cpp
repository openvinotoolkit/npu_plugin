#include "include/mcm/target/keembay/tasks/nce2_task_api.hpp"

mv::Data::TensorIterator mv::createDPUTask(const mv::BaseOpModel& om, mv::Data::OpListIterator opIt, const std::string& name)
{
    //Attrs initializer list must be constructed
    auto attributesKeys = opIt->attrsKeys();
    std::initializer_list<std::pair<std::string, mv::Attribute>> list;

    auto inputs = opIt->getInputTensor();
    auto outputTensor = om.defineOp("VPUTask", inputs, list);
}
