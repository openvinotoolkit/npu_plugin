#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include <iostream>

static void scheduleHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void addressHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void graphfileIndexHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ScheduleHelper)
        .setFunc(scheduleHelperPass)
        .setDescription(
            "Add specific edges for partial serilization"
        );

        MV_REGISTER_PASS(AddressHelper)
        .setFunc(addressHelperPass)
        .setDescription(
            "For debug purposes only: sets addresses of tensors manually"
        );

        MV_REGISTER_PASS(GraphfileIndexHelper)
        .setFunc(graphfileIndexHelperPass)
        .setDescription(
            "For debug purposes only: sets addresses of graphfile tensors manually"
        );
    }

}

void addressHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element &)
{

    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("address_helper_addresses"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No address helper addresses provided");
        return;
    }

    auto addressList = globalParams->get<std::vector<mv::Element>>("address_helper_addresses");
    for (auto e : addressList)
    {
        std::string& name = e.get<std::string>("name_filter");
        std::size_t address = e.get<int>("address");
        pass.log(mv::Logger::MessageType::Debug, "ADDRESS HELPER setting address of "+name+" to "+std::to_string(address));
        try
        {
            auto t = dm.getTensor(name);
            t->setAddress(address);
            auto tensorAllocatorName = t->get<std::set<std::string>>("allocators").begin();
            auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
            mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future
            tensorBufferIt->setOffset(address);
        }
        catch (mv::ArgumentError error)
        {
            pass.log(mv::Logger::MessageType::Debug, error.what());
        }

    }
}

void graphfileIndexHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element &)
{

    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("graphfile_index_helper_addresses"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No address helper addresses provided");
        return;
    }

    auto addressList = globalParams->get<std::vector<mv::Element>>("graphfile_index_helper_addresses");
    for (auto e : addressList)
    {
        std::string& name = e.get<std::string>("name_filter");
        std::size_t address = e.get<std::size_t>("address");
        pass.log(mv::Logger::MessageType::Debug, "ADDRESS HELPER setting address of "+name+" to "+std::to_string(address));
        try
        {
            auto t = dm.getTensor(name);
            t->set<unsigned>("graphFileIndex", address);
        }
        catch (mv::ArgumentError error)
        {
            pass.log(mv::Logger::MessageType::Debug, error.what());
        }

    }
}


void scheduleHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("schedule_helper_edges"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No schedule helper edges provided");
        return;
    }

    auto edgesList = globalParams->get<std::vector<mv::Element>>("schedule_helper_edges");
    for (auto e : edgesList)
    {
        std::string& source = e.get<std::string>("edge_source");
        std::string& sink = e.get<std::string>("edge_sink");
        pass.log(mv::Logger::MessageType::Debug, "SCHEDULE HELPER adding edge from "+source+" to "+sink);
        auto sourceOp = om.getOp(source);
        auto sinkOp = om.getOp(sink);
        cm.defineFlow(sourceOp, sinkOp);
    }
}
