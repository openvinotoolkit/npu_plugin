#include "include/mcm/computation/model/control_model.hpp"
#include "lp_scheduler/remove_redundant_update_barriers.hpp"
#include "lp_scheduler/runtime_simulator.hpp"

static void AssignSafetyBarrierFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);
namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AssignSafetyBarrier)
        .setFunc(AssignSafetyBarrierFcn)
        .setDescription("assign safety barrier");
    }
}

bool splitBarrierCheck(mv::ComputationModel& model)
{
    mv::OpModel om(model);
    auto bOps = om.getOps("BarrierTask");

    for (auto& opIt : bOps)
    {
        auto barrier = opIt->get<mv::Barrier>("Barrier");
        auto consumers = barrier.getConsumers();

        auto baseType = om.getOp(*(consumers.begin()))->getOpType();

        for (auto p = consumers.begin(); p != consumers.end(); ++p)
        {                                  
            auto curType = om.getOp(*(p))->getOpType();
            if((baseType == "DMATask") && (curType != "DMATask"))
            {
                return false;
            }
            if((baseType != "DMATask") && (curType == "DMATask"))
            {
                return false;
            }
        }
    }

    return true;
}

void splitBarrierDependencies(mv::ComputationModel& model, string baseName)
{
    mv::OpModel om(model);
    mv::ControlModel cmodel(model);
    auto bOps = om.getOps("BarrierTask");
    std::vector<std::pair<unsigned short, unsigned short>> dmaContrib(bOps.size());
    auto dmaOps = om.getOps("DMATask");

    for (auto& opIt : dmaOps)
    {
        if (!(opIt->hasAttr("BarrierDeps"))) { continue; }
        auto barrierDeps = opIt->get<mv::BarrierDependencies>("BarrierDeps");
        const std::vector<unsigned>& wait = barrierDeps.getWait();
        const std::vector<unsigned>& update = barrierDeps.getUpdate();
        for (unsigned int j = 0; j < update.size(); ++j)
            ++dmaContrib[update[j]].first;

        for (unsigned int j = 0; j < wait.size(); ++j)
            ++dmaContrib[wait[j]].second;
    }

    // Implementing dependency transformation as described in
    // https://docs.google.com/drawings/d/1WTzFrM8dvFu4ztV4KPekP6X-oNClxup_0gS0sx2nSW4

    // Transformation updated for Dual-DMA setup
    // https://docs.google.com/drawings/d/1nsrTTJO-tHZGpzS6DYh_sOR3yM7L-pR73Ae8DnHhVDU

    std::sort(
            bOps.begin(),
            bOps.end(),
            [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
            );

    int DMA_ENGINES = 1;
    size_t barrier_task_id = 0;
    std::vector<std::pair<short, short> > mappings(bOps.size());
    std::vector<std::string> newBarrierNames;

    for (unsigned short i = 0; i < dmaContrib.size(); ++i)
    {
        mv::Barrier &barrier = bOps[i]->get<mv::Barrier>("Barrier");
        mappings[i] = pair<short, short>(-1, -1);

        unsigned short alpha = dmaContrib[i].first;
        unsigned short beta = static_cast<unsigned short>(barrier.getNumProducers()- alpha);
        unsigned short delta = dmaContrib[i].second;
        unsigned short gamma = static_cast<unsigned short>(barrier.getNumConsumers() - delta);

        if((alpha + beta) == 0)
        {
            throw std::runtime_error("Barrier has no producers");
        }

        if((delta + gamma) == 0)
        {
            throw std::runtime_error("Barrier has no consumers");
        }

        if ((DMA_ENGINES > 1 || beta > 0) &&
                delta > 0)
        {
            // create new barriers
            char barrier_name[64UL]; 
            sprintf(barrier_name, "new_Barrier_%zu", barrier_task_id);

            std::set<std::string> empty_set;
            struct mv::Barrier new_barrier(empty_set, empty_set);

            om.barrierTask(baseName + barrier_name, new_barrier);
            auto barrier_new = om.getOp(baseName + barrier_name);
            assert((barrier_new != om.opEnd()) &&
                    barrier_new->hasAttr("Barrier"));
            mv::Barrier &barrier_barrier_new = barrier_new->get<mv::Barrier>("Barrier");
            barrier_barrier_new.setID(barrier_task_id);
            barrier_barrier_new.setIndex(barrier_barrier_new.getID());
            barrier_barrier_new.setRealBarrierIndex(-1);

            mappings[i].first = static_cast<short>(barrier_task_id);
            barrier_task_id++;
            newBarrierNames.push_back(baseName + barrier_name);
        }

        if (gamma > 0)
        {
            // create new barriers
            char barrier_name[64UL]; 
            sprintf(barrier_name, "new_Barrier_%zu", barrier_task_id);

            std::set<std::string> empty_set;
            struct mv::Barrier new_barrier(empty_set, empty_set);

            om.barrierTask(baseName + barrier_name, new_barrier);
            auto barrier_new = om.getOp(baseName + barrier_name);
            assert((barrier_new != om.opEnd()) &&
                    barrier_new->hasAttr("Barrier"));
            mv::Barrier &barrier_barrier_new = barrier_new->get<mv::Barrier>("Barrier");
            barrier_barrier_new.setID(barrier_task_id);
            barrier_barrier_new.setIndex(barrier_barrier_new.getID());
            barrier_barrier_new.setRealBarrierIndex(-1);

            mappings[i].second = static_cast<short>(barrier_task_id);
            barrier_task_id++;
            newBarrierNames.push_back(baseName + barrier_name);
        }
    }

    for (auto& opIt : dmaOps)
    {
        if (!(opIt->hasAttr("BarrierDeps"))) { continue; }

        mv::BarrierDependencies new_deps;
        auto old_deps = opIt->get<mv::BarrierDependencies>("BarrierDeps");

        const std::vector<unsigned>& old_wait = old_deps.getWait();
        for (auto witr=old_wait.begin(); witr!=old_wait.end(); ++witr) {
        if (mappings[*witr].first >= 0)
        {
            new_deps.addWaitBarrier(static_cast<unsigned short>(mappings[*witr].first));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].first]);
            cmodel.defineFlow(bOp, opIt);
            bOp->get<mv::Barrier>("Barrier").addConsumer(opIt->getName());
        }
        }

        const std::vector<unsigned>& old_update = old_deps.getUpdate();
        for (auto witr=old_update.begin(); witr!=old_update.end(); ++witr) {
        if (DMA_ENGINES > 1)
        {
            if (mappings[*witr].first >= 0)
            {
                new_deps.addUpdateBarrier(static_cast<unsigned short>(mappings[*witr].first));
                auto bOp = om.getOp(newBarrierNames[mappings[*witr].first]);
                cmodel.defineFlow(opIt, bOp);
                bOp->get<mv::Barrier>("Barrier").addProducer(opIt->getName());
            }
        }

        if (mappings[*witr].second >= 0)
        {
            new_deps.addUpdateBarrier(static_cast<unsigned short>(mappings[*witr].second));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].second]);
            cmodel.defineFlow(opIt, bOp);
            bOp->get<mv::Barrier>("Barrier").addProducer(opIt->getName());
        }            
        }

        opIt->set<mv::BarrierDependencies>("BarrierDeps", new_deps);    
    }

    auto dpuOps = om.getOps("DPUTask");
    for (auto& opIt : dpuOps)
    {
        if (!(opIt->hasAttr("BarrierDeps"))) { continue; }

        mv::BarrierDependencies new_deps;
        auto old_deps = opIt->get<mv::BarrierDependencies>("BarrierDeps");

        const std::vector<unsigned>& old_wait = old_deps.getWait();
        for (auto witr=old_wait.begin(); witr!=old_wait.end(); ++witr) {
        if (mappings[*witr].second >= 0)
        {
            new_deps.addWaitBarrier(static_cast<unsigned short>(mappings[*witr].second));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].second]);
            cmodel.defineFlow(bOp, opIt);
            bOp->get<mv::Barrier>("Barrier").addConsumer(opIt->getName());
        }
        }

        const std::vector<unsigned>& old_update = old_deps.getUpdate();
        for (auto witr=old_update.begin(); witr!=old_update.end(); ++witr) {
        if (mappings[*witr].first >= 0)
        {
            new_deps.addUpdateBarrier(static_cast<unsigned short>(mappings[*witr].first));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].first]);
            cmodel.defineFlow(opIt, bOp);
            bOp->get<mv::Barrier>("Barrier").addProducer(opIt->getName());
        }

        if (mappings[*witr].second >= 0)
        {
            new_deps.addUpdateBarrier(static_cast<unsigned short>(mappings[*witr].second));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].second]);
            cmodel.defineFlow(opIt, bOp);
            bOp->get<mv::Barrier>("Barrier").addProducer(opIt->getName());
        }            
        }

        opIt->set<mv::BarrierDependencies>("BarrierDeps", new_deps);    
    }

    auto upaOps = om.getOps("UPATask");
    for (auto& opIt : upaOps)
    {
        if (!(opIt->hasAttr("BarrierDeps"))) { continue; }

        mv::BarrierDependencies new_deps;
        auto old_deps = opIt->get<mv::BarrierDependencies>("BarrierDeps");

        const std::vector<unsigned>& old_wait = old_deps.getWait();
        for (auto witr=old_wait.begin(); witr!=old_wait.end(); ++witr) {
        if (mappings[*witr].second >= 0)
        {
            new_deps.addWaitBarrier(static_cast<unsigned short>(mappings[*witr].second));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].second]);
            cmodel.defineFlow(bOp, opIt);
            bOp->get<mv::Barrier>("Barrier").addConsumer(opIt->getName());
        }
        }

        const std::vector<unsigned>& old_update = old_deps.getUpdate();
        for (auto witr=old_update.begin(); witr!=old_update.end(); ++witr) {
        if (mappings[*witr].first >= 0)
        {
            new_deps.addUpdateBarrier(static_cast<unsigned short>(mappings[*witr].first));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].first]);
            cmodel.defineFlow(opIt, bOp);
            bOp->get<mv::Barrier>("Barrier").addProducer(opIt->getName());
        }

        if (mappings[*witr].second >= 0)
        {
            new_deps.addUpdateBarrier(static_cast<unsigned short>(mappings[*witr].second));
            auto bOp = om.getOp(newBarrierNames[mappings[*witr].second]);
            cmodel.defineFlow(opIt, bOp);
            bOp->get<mv::Barrier>("Barrier").addProducer(opIt->getName());
        }            
        }

        opIt->set<mv::BarrierDependencies>("BarrierDeps", new_deps);    
    }

    for (auto& opIt : bOps)
    {
        om.removeOp(opIt);
    }
}

bool checkCircleWithNewBarrier(mv::ControlModel& cm, mv::Data::OpListIterator& safetyBarrierOp, mv::Barrier& newBarrier)
{
    bool pass = true;
    auto consumers = newBarrier.getConsumers();
    auto producers = newBarrier.getProducers();

    for (auto p = consumers.begin(); p != consumers.end(); ++p)
    {
        if ((cm.pathExists(cm.switchContext(cm.getOp(*p)), cm.switchContext(safetyBarrierOp))))
        {
            pass = false;
            return pass;
        }
    }

    for (auto p = producers.begin(); p != producers.end(); ++p)
    {
        if ((cm.pathExists(cm.switchContext(cm.getOp(*p)), cm.switchContext(safetyBarrierOp))))
        {
            pass = false;
            return pass;
        }
    }

    return pass;
}

bool checkCommonConsumer(mv::ControlModel& cm, mv::Data::OpListIterator& oldBarrierOp, mv::Data::OpListIterator& safetyBarrier)
{
    bool pass = true;
    auto preVIDConsumers = (oldBarrierOp->get<mv::Barrier>("Barrier")).getConsumers();

    for (auto p = preVIDConsumers.begin(); p != preVIDConsumers.end(); ++p)
    {
        if(cm.pathExists(cm.switchContext(safetyBarrier), cm.switchContext(cm.getOp(*p))))
        {
            pass = false;
            return pass;
        }
    }

    return pass;
}

bool checkTimeConstraint(mv::ControlModel& cm, std::vector<mv::Data::OpListIterator>& barrierTasks, int safeID, int currentID)
{
    if(barrierTasks[safeID]->hasAttr("preVID"))
    {
        auto preSafeBarrier = barrierTasks[safeID]->get<unsigned>("preVID");
        auto oldVirtualID = barrierTasks[currentID]->get<unsigned>("preVID");
        if(barrierTasks[preSafeBarrier]->get<unsigned>("readyForReset") > barrierTasks[oldVirtualID]->get<unsigned>("readyForReset"))
        {
            return false;
        }

        auto preSafeBarrierConsumers = (barrierTasks[preSafeBarrier]->get<mv::Barrier>("Barrier")).getConsumers();
        auto oldVIDConsumers = (barrierTasks[oldVirtualID]->get<mv::Barrier>("Barrier")).getConsumers();

        // bool barrierResetCheck = true;
        for(auto& p1: preSafeBarrierConsumers)
        {
            bool depsCheck = false; 
            for(auto& p2: oldVIDConsumers)
            {
                if((cm.pathExists(cm.switchContext(cm.getOp(p1)), cm.switchContext(cm.getOp(p2)))))
                {
                    depsCheck = true;
                    break;
                }
            }

            if(!depsCheck)
            {
                return false;
            }
        }

        for(auto& p1: oldVIDConsumers)
        {
            bool depsCheck = false; 
            for(auto& p2: barrierTasks[safeID]->get<mv::Barrier>("Barrier").getConsumers())
            {
                if((cm.pathExists(cm.switchContext(cm.getOp(p1)), cm.switchContext(cm.getOp(p2)))))
                {
                    depsCheck = true;
                    break;
                }
            }

            if(!depsCheck)
            {
                return false;
            }
        }
    }

    return true;  
}

bool opTypeCheck(const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b)
{
    if((a->getOpType() == "DMATask") &&  (b->getOpType() != "DMATask"))
        return false;
    if((a->getOpType() != "DMATask") &&  (b->getOpType() == "DMATask"))
        return false;
    return true;
}

int countNewEdge(mv::ControlModel& cm, mv::Data::OpListIterator& safetyBarrierOp, mv::Data::OpListIterator& barrierOp,
                 bool isConsumer, bool& foundSafetyBarrier, int oldVirtualID, std::vector<mv::Control::OpListIterator>& targetOps,
                 std::vector<mv::Control::OpListIterator>& sortedDMAOps, std::vector<mv::Control::OpListIterator>& sortedUPAOps,
                 std::vector<mv::Control::OpListIterator>& sortedDPUOps, std::vector<mv::Data::OpListIterator>& barrierTasks,
                 bool& splitCheck)
{
    mv::OpModel om(cm);
    int addControl= 0;
    auto newBarrier= barrierOp->get<mv::Barrier>("Barrier");
    unsigned virtualID = newBarrier.getIndex();
    auto tasks = newBarrier.getProducers();
    auto safetyBarrierID = (safetyBarrierOp->get<mv::Barrier>("Barrier")).getIndex();
    if (isConsumer)
    {
        tasks = newBarrier.getConsumers();
    }

    auto preBarrier = barrierTasks[oldVirtualID]->get<mv::Barrier>("Barrier");
    auto preConsumers = preBarrier.getConsumers();

    std::vector<std::string> removeRedundantTasks;

    // if(!isConsumer)
    // {
    //     for (auto p = tasks.begin(); p != tasks.end(); ++p)
    //     {
    //         bool hasDependency = true;
    //         for (auto pCons = preConsumers.begin(); pCons != preConsumers.end(); ++pCons)
    //         {
    //             if(!cm.pathExists(cm.switchContext(cm.getOp(*pCons)), cm.switchContext(cm.getOp(*p))))
    //             {
    //                 hasDependency = false;
    //                 break;
    //             }
    //         }

    //         if(!hasDependency)
    //             removeRedundantTasks.push_back(*p);
    //     }
    // }
    // else
    // {
    //     for (auto p = tasks.begin(); p != tasks.end(); ++p)
    //     {
    //         bool hasDependency = true;
    //         for (auto pCons = preConsumers.begin(); pCons != preConsumers.end(); ++pCons)
    //         {
    //             bool checkOnePairDependency = false;
    //             auto barrierDepsWait = cm.getOp(*p)->get<mv::BarrierDependencies>("BarrierDeps").getWait();
    //             for(int waitID = 0; !checkOnePairDependency && (waitID < barrierDepsWait.size()); waitID++)
    //             {
    //                 if(barrierDepsWait[waitID] != virtualID)
    //                 {
    //                 if(cm.pathExists(cm.switchContext(cm.getOp(*pCons)), cm.switchContext(barrierTasks[barrierDepsWait[waitID]])))
    //                 {
    //                     checkOnePairDependency = true;
    //                     break;
    //                 }
    //                 }
    //             }

    //             if(!checkOnePairDependency && (cm.getOp(*p)->getOpType() == "DMATask"))
    //             {
    //                 auto iter = std::find(sortedDMAOps.begin(), sortedDMAOps.end(), cm.switchContext(cm.getOp(*p)));
    //                 if(iter != sortedDMAOps.begin())
    //                 {
    //                     iter--;
    //                     if(cm.pathExists(cm.switchContext(cm.getOp(*pCons)), (*iter)))
    //                     {
    //                         checkOnePairDependency = true;
    //                     }
    //                 }
    //             }
    //             else if(!checkOnePairDependency && (cm.getOp(*p)->getOpType() == "DPUTask"))
    //             {
    //                 auto iter = std::find(sortedDPUOps.begin(), sortedDPUOps.end(), cm.switchContext(cm.getOp(*p)));
    //                 if(iter != sortedDPUOps.begin())
    //                 {
    //                     iter--;
    //                     if(cm.pathExists(cm.switchContext(cm.getOp(*pCons)), (*iter)))
    //                     {
    //                         checkOnePairDependency = true;
    //                     }
    //                 }
    //             }
    //             else if(!checkOnePairDependency && (cm.getOp(*p)->getOpType() == "UPATask"))
    //             {
    //                 auto iter = std::find(sortedUPAOps.begin(), sortedUPAOps.end(), cm.switchContext(cm.getOp(*p)));
    //                 if(iter != sortedUPAOps.begin())
    //                 {
    //                     iter--;
    //                     if(cm.pathExists(cm.switchContext(cm.getOp(*pCons)), (*iter)))
    //                     {
    //                         checkOnePairDependency = true;
    //                     }
    //                 }
    //             }

    //             if(!checkOnePairDependency)
    //             {
    //                 hasDependency = false;
    //                 break;
    //             }
    //         }

    //         if(!hasDependency)
    //             removeRedundantTasks.push_back(*p);
    //     }
    // }

    for (auto p = tasks.begin(); p != tasks.end(); ++p)
    {
        // bool hasDependency = true;
        auto baseType = cm.getOp(*(preConsumers.begin()))->getOpType();
        bool sameType = true;
        for (auto pCons = preConsumers.begin(); pCons != preConsumers.end(); ++pCons)
        {
            if((cm.getOp(*pCons))->getOpType() != baseType)
            {
                sameType = false;
                break;
            }
        }

        // check DMA Task distance
        if(sameType)
        {
            if(cm.getOp(*p)->getOpType() == baseType)
            {
                if(baseType == "DMATask")
                {
                    int minDistance = std::numeric_limits<int>::max();
                    auto iter1 = std::find(sortedDMAOps.begin(), sortedDMAOps.end(), cm.switchContext(cm.getOp(*p)));

                    for (auto pCons1 = preConsumers.begin(); pCons1 != preConsumers.end(); ++pCons1)
                    {
                        // std::cout << *pCons << std::endl;
                        auto iter0 = std::find(sortedDMAOps.begin(), sortedDMAOps.end(), cm.switchContext(cm.getOp(*pCons1)));
                        minDistance = std::min(minDistance, (int)(iter1 - iter0));
                    }                                

                    if(minDistance < 2)
                    {
                        std::cout << "from " << *(preConsumers.begin()) << "to " << *p << ", distance " << minDistance << std::endl;
                        removeRedundantTasks.push_back(*p);
                    }
                }

                // if(baseType == "DPUTask")
                // {
                //     int minDistance = std::numeric_limits<int>::max();
                //     auto iter1 = std::find(sortedDPUOps.begin(), sortedDPUOps.end(), cm.switchContext(cm.getOp(*p)));

                //     for (auto pCons1 = preConsumers.begin(); pCons1 != preConsumers.end(); ++pCons1)
                //     {
                //         // std::cout << *pCons << std::endl;
                //         auto iter0 = std::find(sortedDPUOps.begin(), sortedDPUOps.end(), cm.switchContext(cm.getOp(*pCons1)));
                //         minDistance = std::min(minDistance, (int)(iter1 - iter0));
                //     }                                

                //     // TODO: make decision depends on the estimation of the data amount for DMA Task 
                //     if(minDistance < 2)
                //     {
                //         std::cout << "from " << *(preConsumers.begin()) << "to " << *p << ", distance " << minDistance << std::endl;
                //         removeRedundantTasks.push_back(*p);
                //     }
                // }         

                // if(baseType == "UPATask")
                // {
                //     int minDistance = std::numeric_limits<int>::max();
                //     auto iter1 = std::find(sortedUPAOps.begin(), sortedUPAOps.end(), cm.switchContext(cm.getOp(*p)));

                //     for (auto pCons1 = preConsumers.begin(); pCons1 != preConsumers.end(); ++pCons1)
                //     {
                //         // std::cout << *pCons << std::endl;
                //         auto iter0 = std::find(sortedUPAOps.begin(), sortedUPAOps.end(), cm.switchContext(cm.getOp(*pCons1)));
                //         minDistance = std::min(minDistance, (int)(iter1 - iter0));
                //     }                                

                //     // TODO: make decision depends on the estimation of the data amount for DMA Task 
                //     if(minDistance < 2)
                //     {
                //         std::cout << "from " << *(preConsumers.begin()) << "to " << *p << ", distance " << minDistance << std::endl;
                //         removeRedundantTasks.push_back(*p);
                //     }
                // }                      
            }
            else
            {
                removeRedundantTasks.push_back(*p);
            }            
        }
        else
            removeRedundantTasks.push_back(*p);   
    }   

    auto oldConsumers = (safetyBarrierOp->get<mv::Barrier>("Barrier")).getConsumers();

    unsigned minSID = std::numeric_limits<unsigned>::max();
    for (auto oc = oldConsumers.begin(); (oc != oldConsumers.end()); ++oc)
    {
        auto op = cm.getOp(*oc);
        minSID = std::min(op->get<unsigned>("scheduleID"), minSID);
    }

    for (auto p = removeRedundantTasks.begin(); p != removeRedundantTasks.end(); ++p)
    {
        bool hasDependency = false;
        // check if dependency exists
        if (isConsumer)
        {
            for (auto oc = oldConsumers.begin(); (oc != oldConsumers.end()); ++oc)
            {
                auto op = cm.getOp(*oc);
                if((op->getOpType() == cm.getOp(*p)->getOpType()) &&
                   (cm.pathExists(cm.switchContext(op), cm.switchContext(cm.getOp(*p)))))
                {
                    hasDependency = true;
                    break;
                }
            }

            auto barrierDepsWait = cm.getOp(*p)->get<mv::BarrierDependencies>("BarrierDeps").getWait();
            for(unsigned waitID = 0; !hasDependency && (waitID < barrierDepsWait.size()); waitID++)
            {
                if(barrierDepsWait[waitID] != virtualID)
                {
                   if(cm.pathExists(cm.switchContext(safetyBarrierOp), cm.switchContext(barrierTasks[barrierDepsWait[waitID]])))
                   {
                       hasDependency = true;
                       break;
                   }
                }
            }

            if(!hasDependency && (cm.getOp(*p)->getOpType() == "DMATask"))
            {
                auto iter = std::find(sortedDMAOps.begin(), sortedDMAOps.end(), cm.switchContext(cm.getOp(*p)));
                if(iter != sortedDMAOps.begin())
                {
                    iter--;
                    if(cm.pathExists(cm.switchContext(safetyBarrierOp), (*iter)))
                    {
                        hasDependency = true;
                    }
                }
            }
            else if(!hasDependency && (cm.getOp(*p)->getOpType() == "DPUTask"))
            {
                auto iter = std::find(sortedDPUOps.begin(), sortedDPUOps.end(), cm.switchContext(cm.getOp(*p)));
                if(iter != sortedDPUOps.begin())
                {
                    iter--;
                    if(cm.pathExists(cm.switchContext(safetyBarrierOp), (*iter)))
                    {
                        hasDependency = true;
                    }
                }
            }
            else if(!hasDependency && (cm.getOp(*p)->getOpType() == "UPATask"))
            {
                auto iter = std::find(sortedUPAOps.begin(), sortedUPAOps.end(), cm.switchContext(cm.getOp(*p)));
                if(iter != sortedUPAOps.begin())
                {
                    iter--;
                    if(cm.pathExists(cm.switchContext(safetyBarrierOp), (*iter)))
                    {
                        hasDependency = true;
                    }
                }
            }
        }
        else
        {
            // for producer
            if(cm.pathExists(cm.switchContext(safetyBarrierOp), cm.switchContext(cm.getOp(*p))))
            {
                hasDependency = true;
            }
        }

        auto waitList = cm.getOp(*p)->get<mv::BarrierDependencies>("BarrierDeps").getWait();
        if(std::find(waitList.begin(), waitList.end(), oldVirtualID) != waitList.end())
        {
            foundSafetyBarrier = false;
        }
        if(std::find(waitList.begin(), waitList.end(), safetyBarrierID) != waitList.end())
        {
            hasDependency = true;
        }

        if(!foundSafetyBarrier)
        {
            break;
        }        
        if (hasDependency)
        {
            continue;
        }

        if(!opTypeCheck(cm.getOp(*(oldConsumers.begin())), cm.getOp(*p)))
        {
            splitCheck = false;
            // if(cm.getOp(*p)->getOpType() == "DMATask")
            // {
                // auto oldProducers = (safetyBarrierOp->get<mv::Barrier>("Barrier")).getProducers();
                // foundSafetyBarrier = false;
                // for (auto p = oldProducers.begin(); p != oldProducers.end(); ++p)
                // {
                //     if(cm.getOp(*p)->getOpType() != "DMATask")
                //     {
                //         foundSafetyBarrier = true;
                //         break;
                //     }
                // }
            // }
        }
        if(!foundSafetyBarrier)
        {
            break;
        }  

        // add new dependency
        // if(safetyBarrierOp->hasAttr("lastB"))
        // {
        //     foundSafetyBarrier = true;
        //     if(std::find(targetOps.begin(), targetOps.end(), cm.switchContext(cm.getOp(*p))) == targetOps.end())
        //     {
        //         targetOps.push_back(cm.switchContext(cm.getOp(*p)));
        //         addControl++;
        //     }
        //     continue;
        // }

        auto readyForConsume = safetyBarrierOp->get<unsigned>("readyForConsume");
        // auto readyForReset = safetyBarrierOp->get<unsigned>("readyForReset");
        unsigned readyForReset = std::numeric_limits<int>::max();
        auto opTime = cm.getOp(*p)->get<unsigned>("scheduleID");

        if(((opTime >= readyForConsume) && (opTime <= readyForReset) && foundSafetyBarrier && (opTime >= barrierOp->get<unsigned>("scheduleID"))))
        {
            if(std::find(targetOps.begin(), targetOps.end(), cm.switchContext(cm.getOp(*p))) == targetOps.end())
            {
                targetOps.push_back(cm.switchContext(cm.getOp(*p)));
                addControl++;
            }
        }
        // check the last task
        else if(opTime > readyForReset)
        {
            foundSafetyBarrier = false;
            auto baseType = cm.getOp(*p)->getOpType();
            if(baseType == "DMATask")
            {
                auto iter = std::find(sortedDMAOps.rbegin(), sortedDMAOps.rend(), cm.switchContext(cm.getOp(*p)));
                if(iter != sortedDMAOps.rend())
                {
                    iter++;
                    for(; (iter!=sortedDMAOps.rend()) && (!foundSafetyBarrier); iter++)
                    {
                        if ((cm.pathExists((*iter), cm.switchContext(safetyBarrierOp))))
                        {
                            // continue;
                        }
                        else
                        {
                            opTime = (*iter)->get<unsigned>("scheduleID");
                            if((opTime >= readyForConsume) && (opTime <= readyForReset) && (opTime >= barrierOp->get<unsigned>("scheduleID")))
                            {
                                foundSafetyBarrier = true;
                                auto waitListTmp = (*iter)->get<mv::BarrierDependencies>("BarrierDeps").getWait();
                                if(std::find(waitListTmp.begin(), waitListTmp.end(), oldVirtualID) != waitListTmp.end())
                                {
                                    foundSafetyBarrier = false;
                                }
                                else
                                {
                                    if(std::find(targetOps.begin(), targetOps.end(), *iter) == targetOps.end())
                                    {
                                        targetOps.push_back(*iter);
                                        addControl++;
                                    }
                                }
                            }
                            else if(opTime < readyForConsume)
                            {
                                break;
                            }
                        }
                    }
                }
            }

            if(baseType == "DPUTask")
            {
                auto iter = std::find(sortedDPUOps.rbegin(), sortedDPUOps.rend(), cm.switchContext(cm.getOp(*p)));
                if(iter != sortedDPUOps.rend())
                {
                    iter++;
                    for(; (iter!=sortedDPUOps.rend()) && (!foundSafetyBarrier); iter++)
                    {
                        if ((cm.pathExists((*iter), cm.switchContext(safetyBarrierOp))))
                        {
                            // continue;
                        }
                        else
                        {
                            opTime = (*iter)->get<unsigned>("scheduleID");
                            if((opTime >= readyForConsume) && (opTime <= readyForReset) && (opTime >= barrierOp->get<unsigned>("scheduleID")))
                            {
                                foundSafetyBarrier = true;
                                auto waitListTmp = (*iter)->get<mv::BarrierDependencies>("BarrierDeps").getWait();
                                if(std::find(waitListTmp.begin(), waitListTmp.end(), oldVirtualID) != waitListTmp.end())
                                {
                                    foundSafetyBarrier = false;
                                }
                                else
                                {
                                    if(std::find(targetOps.begin(), targetOps.end(), *iter) == targetOps.end())
                                    {
                                        targetOps.push_back(*iter);
                                        addControl++;
                                    }
                                }
                            }
                            else if(opTime < readyForConsume)
                            {
                                break;
                            }
                        }
                    }
                }
            }

            if(baseType == "UPATask")
            {
                auto iter = std::find(sortedUPAOps.rbegin(), sortedUPAOps.rend(), cm.switchContext(cm.getOp(*p)));
                if(iter != sortedUPAOps.rend())
                {
                    iter++;
                    for(; (iter!=sortedUPAOps.rend()) && (!foundSafetyBarrier); iter++)
                    {
                        if ((cm.pathExists((*iter), cm.switchContext(safetyBarrierOp))))
                        {
                            // continue;
                        }
                        else
                        {
                            opTime = (*iter)->get<unsigned>("scheduleID");
                            if((opTime >= readyForConsume) && (opTime <= readyForReset) && (opTime >= barrierOp->get<unsigned>("scheduleID")))
                            {
                                foundSafetyBarrier = true;
                                auto waitListTmp = (*iter)->get<mv::BarrierDependencies>("BarrierDeps").getWait();
                                if(std::find(waitListTmp.begin(), waitListTmp.end(), oldVirtualID) != waitListTmp.end())
                                {
                                    foundSafetyBarrier = false;
                                }
                                else
                                {
                                    if(std::find(targetOps.begin(), targetOps.end(), *iter) == targetOps.end())
                                    {
                                        targetOps.push_back(*iter);
                                        addControl++;
                                    }
                                }
                            }
                            else if(opTime < readyForConsume)
                            {
                                break;
                            }
                        }
                    }
                }
            }
        }
        else{
            foundSafetyBarrier = false;
        }
    }

    return addControl;
}

void addNewEdge(mv::ControlModel& cm, mv::Data::OpListIterator& safetyBarrierOp, std::vector<mv::Control::OpListIterator>& targetOps)
{
    for (auto iter= targetOps.begin(); iter!= targetOps.end(); iter++)
    {
        auto& dstBarrier = (*iter)->get<mv::BarrierDependencies>("BarrierDeps");
        {
            std::cout << "add dependency from " << safetyBarrierOp->getName() << " to " << (*iter)->getName() << std::endl;                               
            dstBarrier.addWaitBarrier(safetyBarrierOp->get<mv::Barrier>("Barrier").getIndex());
            safetyBarrierOp->get<mv::Barrier>("Barrier").addConsumer((*iter)->getName());
            cm.defineFlow(safetyBarrierOp, cm.getOp((*iter)->getName()));
        }
    }
}

std::string addNewBarrier(mv::ControlModel& cm, mv::Data::OpListIterator& safetyBarrierOp, std::vector<mv::Control::OpListIterator>& targetOps, int size, int& maxWaitID,
                          std::vector<mv::Control::OpListIterator>& sortedDMAOps)
{
    auto consumers = safetyBarrierOp->get<mv::Barrier>("Barrier").getConsumers();

    // create new barrier
    mv::OpModel om(cm);
    static size_t new_safe_barrier_task_id=0UL;
    
    char barrier_name[64UL]; 
    sprintf(barrier_name, "Safe_Barrier_%zu", new_safe_barrier_task_id++);
    std::set<std::string> empty_set;
    struct mv::Barrier new_barrier(empty_set, empty_set);
    static int newSafeBarrierCount = 0;

    om.barrierTask(barrier_name, new_barrier);
    auto barrier_new = om.getOp(barrier_name);
    mv::Barrier &barrier = barrier_new->get<mv::Barrier>("Barrier");
    barrier.setID(size);
    barrier.setIndex(barrier.getID());   
    newSafeBarrierCount++;

    // std::cout << "create new barrier done" << std::endl;

    for (auto iter= targetOps.begin(); iter!= targetOps.end(); iter++)
    {
        auto& dstBarrier = (*iter)->get<mv::BarrierDependencies>("BarrierDeps");
        auto waitList = dstBarrier.getWait();
        if(waitList.size())
        {
            // std::cout << "get waitList" << waitList.size() << std::endl;
            auto localMax = std::max_element(waitList.begin(), waitList.end());
            // std::cout << "get localMax " << *localMax << std::endl;
            maxWaitID = std::max(maxWaitID, (int)(*localMax));
            // std::cout << "assign max" << std::endl;
        }
    }

    if(maxWaitID < 0)
    {
        auto iter = std::find(sortedDMAOps.begin(), sortedDMAOps.end(), (*(targetOps.begin())));
        while(iter != sortedDMAOps.begin())
        {
            iter--;
            auto& dstBarrier = (*iter)->get<mv::BarrierDependencies>("BarrierDeps");
            auto waitList = dstBarrier.getWait();
            if(waitList.size() && (!(*iter)->hasAttr("addNewBarrier")))
            {
                auto localMax = std::max_element(waitList.begin(), waitList.end());
                maxWaitID = std::max(maxWaitID, (int)(*localMax));
                break;
            }
        }
    }

    // std::cout << "get maxWaitID" << std::endl;

    for (auto iter= targetOps.begin(); iter!= targetOps.end(); iter++)
    {
        std::cout << "add dependency from " << barrier_new->getName() << " to " << (*iter)->getName() << std::endl;
        auto& dstBarrier = (*iter)->get<mv::BarrierDependencies>("BarrierDeps");
        {                               
            dstBarrier.addWaitBarrier(barrier_new->get<mv::Barrier>("Barrier").getIndex());
            barrier_new->get<mv::Barrier>("Barrier").addConsumer((*iter)->getName());
            cm.defineFlow(barrier_new, cm.getOp((*iter)->getName()));
        }
        (*iter)->set<bool>("addNewBarrier", true);
    }

    for(auto& p: consumers)
    {
        auto& dstBarrier = cm.getOp(p)->get<mv::BarrierDependencies>("BarrierDeps");
        {                               
            dstBarrier.addUpdateBarrier(barrier_new->get<mv::Barrier>("Barrier").getIndex());
            barrier_new->get<mv::Barrier>("Barrier").addProducer(p);
            cm.defineFlow(cm.getOp(p), barrier_new);
        }
    }

    return barrier_new->getName();
}

bool is_barrier_op(mv::Control::OpListIterator op)
{
    return op->getOpType() == "BarrierTask";
}

void updateBarrierDependency(mv::ControlModel& cm)
{
    // STEP-1: clear all the references //
    mv::Control::FlowListIterator fitr, fitr_next;
    for (fitr=cm.flowBegin(); fitr!=cm.flowEnd(); ++fitr) {
        mv::Control::OpListIterator src_itr = fitr.source();
        mv::Control::OpListIterator sink_itr = fitr.sink(); 
        mv::Control::OpListIterator bar_itr, op_itr;

        assert( (src_itr != cm.opEnd()) && (sink_itr != cm.opEnd()) );
        if (!is_barrier_op(src_itr) && !is_barrier_op(sink_itr)) { continue; }

        if (is_barrier_op(src_itr)) {
        assert(!is_barrier_op(sink_itr));
        bar_itr = src_itr;
        op_itr = sink_itr;
        } else {
        assert(is_barrier_op(sink_itr));
        bar_itr = sink_itr;
        op_itr = src_itr;
        }

    
        mv::Barrier& barrier = bar_itr->get<mv::Barrier>("Barrier");
        barrier.clearProducersConsumers();

        mv::BarrierDependencies& barrierRef =
        op_itr->get<mv::BarrierDependencies>("BarrierDeps");
        barrierRef.clear();
    }

    // STEP-2:
    // foreach control edge (u, v) 
    //
    // CASE-1: (bar->op)
    //    op.addWaitBarrier(bar)
    //    bar.addConsumer(op) 
    //   
    // CASE-2: (op->bar)
    for (fitr=cm.flowBegin(); fitr!=cm.flowEnd(); ++fitr) {
        mv::Control::OpListIterator src_itr = fitr.source();
        mv::Control::OpListIterator sink_itr = fitr.sink(); 
        assert( (src_itr != cm.opEnd()) && (sink_itr != cm.opEnd()) );

        if (!is_barrier_op(src_itr) && !is_barrier_op(sink_itr)) { continue; }

        if (is_barrier_op(src_itr)) {
        assert(!is_barrier_op(sink_itr));

        mv::Barrier& barrier = src_itr->get<mv::Barrier>("Barrier");
        mv::BarrierDependencies& barrierRef =
            sink_itr->get<mv::BarrierDependencies>("BarrierDeps");


        barrierRef.addWaitBarrier(barrier.getIndex());
        barrier.addConsumer(sink_itr->getName());
        } else {
        assert(!is_barrier_op(src_itr));

        mv::Barrier& barrier = sink_itr->get<mv::Barrier>("Barrier");
        mv::BarrierDependencies& barrierRef =
            src_itr->get<mv::BarrierDependencies>("BarrierDeps");


        barrierRef.addUpdateBarrier(barrier.getIndex());
        barrier.addProducer(src_itr->getName());
        }
    } // foreach control edge //
}

bool barrierSafety(mv::ComputationModel& model, size_t physicalBarrierBound, size_t iteration, int specialUPABarrier, std::string referenceDevice)
{
    // bool needUpdate = false;
    mv::OpModel om(model);
    mv::ControlModel cm(model); 
    auto allOps = om.getOps();
    auto barrierTasks = om.getOps("BarrierTask");
    std::vector<mv::Data::OpListIterator> newBarrierTasks;
    std::vector<unsigned> newBarrierTasksID;
    std::sort(
        barrierTasks.begin(),
        barrierTasks.end(),
        [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
        );

    std::vector<int> toVirtual(physicalBarrierBound, -1);
    unsigned count = 0;
    auto sortedDMAOps = cm.schedulingSortDMA();
    auto sortedOps = cm.schedulingSortDPUorUPA();
    vector<mv::Control::OpListIterator>  sortedDPUOps, sortedUPAOps;
    for(unsigned i = 0; i < (sortedOps.size()); i++)
    {
        if(sortedOps[i]->getOpType() == "DPUTask")
            sortedDPUOps.push_back(sortedOps[i]);
        else
        {
            sortedUPAOps.push_back(sortedOps[i]);
        }        
    }

    // add implicit dependency
    if(!iteration)
    {
        // add dpu dependency
        for(unsigned i = 0; i < (sortedDPUOps.size() - 1); i++)
        {
            cm.defineFlow(sortedDPUOps[i], sortedDPUOps[i+1]);
        }

        // add upa dependency
        if(sortedUPAOps.size() > 1)
        {
            for(unsigned i = 0; i < (sortedUPAOps.size() - 1); i++)
            {
                auto barrierDeps = sortedUPAOps[i+1]->get<mv::BarrierDependencies>("BarrierDeps");
                if(barrierDeps.getWaitSize() > 0)
                    cm.defineFlow(sortedUPAOps[i], sortedUPAOps[i+1]);
            }
        }

        // add dma dependency
        for(unsigned i = 0; i < (sortedDMAOps.size() - 1); i++)
        {
            cm.defineFlow(sortedDMAOps[i], sortedDMAOps[i+1]);
        }
    }

    for(auto& barrierOp: barrierTasks)
    {
        if(barrierOp->hasAttr("safetyBarrierCount"))
            barrierOp->erase("safetyBarrierCount");
        if(barrierOp->hasAttr("BarrierDeps"))
            barrierOp->erase("BarrierDeps");
    }

    for(auto& it : allOps)
    {
        if(it->hasAttr("addNewBarrier"))
            it->erase("addNewBarrier");
        // if(it->hasAttr("schedulingNumber"))
        //     it->set<unsigned>("scheduleID", it->get<unsigned>("schedulingNumber"));
    }

    for(auto& barrierOp: barrierTasks)
    {
        // auto scheduleID = barrierOp->get<unsigned>("scheduleID");
        auto barrier = barrierOp->get<mv::Barrier>("Barrier");
        auto virtualID = barrier.getIndex();
        auto physicalID = barrier.getRealBarrierIndex(); 
        if(toVirtual[physicalID] >= 0)
        {
            barrierOp->set<unsigned>("preVID", toVirtual[physicalID]);
            barrierOp->set<unsigned>("scheduleID", barrierTasks[toVirtual[physicalID]]->get<unsigned>("readyForReset"));
        }
        toVirtual[physicalID] = virtualID;
    }

    vector<unsigned> lastBarrierPool;
    for(unsigned i = 0; i < toVirtual.size(); i++)
    {
        auto vid = toVirtual[i];
        if(vid >= 0)
        {
            barrierTasks[vid]->set<bool>("lastB", true);
            barrierTasks[vid]->set<unsigned>("readyForReset", std::numeric_limits<unsigned>::max());
            if((!specialUPABarrier) || (i < (physicalBarrierBound - 1)))
                lastBarrierPool.push_back(vid);
        }
        toVirtual[i] = -1;
    }
    // auto lastSafetyID = 0;
    auto barrierTasksNum = barrierTasks.size();

    // core logic, find safe barrier for each barrier
    for(auto& barrierOp: barrierTasks)
    {
        auto barrier = barrierOp->get<mv::Barrier>("Barrier");
        unsigned virtualID = barrier.getIndex();
        unsigned physicalID = barrier.getRealBarrierIndex();        

        if((toVirtual[physicalID] >= 0))
        {
            auto oldVirtualID = toVirtual[physicalID];
            auto consumers = barrier.getConsumers();
            auto producers = barrier.getProducers();
            bool foundSafetyBarrier = false;

            std::vector<unsigned> safetyBarrierPool = barrierOp->get<std::vector<unsigned>>("safetyBarrierPool");
            // WA for safety barrier assignment of special UPABarrier, need to remove runtime simulator
            if(specialUPABarrier && (physicalID == (physicalBarrierBound - 1)))
            {
                safetyBarrierPool.clear();
                for(unsigned i = oldVirtualID + 1; i < virtualID; i++)
                {
                    safetyBarrierPool.push_back(i);
                }
            }
            // else
            // {
            //     for(int i = oldVirtualID + 1; i < virtualID; i++)
            //     {
            //         safetyBarrierPool.push_back(i);
            //     }
            // }
            
            safetyBarrierPool.insert(safetyBarrierPool.end(), lastBarrierPool.begin(), lastBarrierPool.end());
            std::sort(safetyBarrierPool.begin(), safetyBarrierPool.end());
            safetyBarrierPool.erase(unique(safetyBarrierPool.begin(), safetyBarrierPool.end()), safetyBarrierPool.end());
            // std::cout << "insert lastBarrierPool" << std::endl;

            int filterdID = -1;
            std::vector<int> filterdIDPool;
            std::vector<int> zeroDepPool;
            std::vector<int> splitBarrierPool;
            std::vector<int> addSameTypeTaskPool;
            std::map<unsigned, vector<mv::Control::OpListIterator>> res_targetOps;
            unsigned minAdd = std::numeric_limits<unsigned>::max();
            unsigned minAddSplit = std::numeric_limits<unsigned>::max();
            unsigned minAddSplitDPU = std::numeric_limits<unsigned>::max();
            // unsigned globalNewConsumersMax = std::numeric_limits<unsigned>::max();
            // check each candidate safety barrier in pool
            for(unsigned i = 0; (i < safetyBarrierPool.size()); i++)
            {
                // std::cout << "try VID: " << safetyBarrierPool[i] << std::endl;
                bool splitCheck = true;
                bool addDifferentTypeOfTask = false;
                foundSafetyBarrier = true;
                auto srcUpdateBarrier = safetyBarrierPool[i];
               
                // check safety barrier is valid or not
                if(srcUpdateBarrier >= virtualID)
                    continue;

                if(barrierTasks[srcUpdateBarrier]->hasAttr("upaConsumer"))
                    continue;

                // if(!barrierTasks[srcUpdateBarrier]->hasAttr("lastB"))
                {
                    if(barrierOp->get<unsigned>("scheduleID") > barrierTasks[srcUpdateBarrier]->get<unsigned>("readyForConsume"))
                    {
                        // std::cout << "readyForConsume fail" << std::endl;
                        foundSafetyBarrier = false;
                        continue;
                    }
                }

                auto safeBarrierProds = barrierTasks[srcUpdateBarrier]->get<mv::Barrier>("Barrier").getProducers();
                int workload = 0;
                for(auto p = safeBarrierProds.begin(); p != safeBarrierProds.end(); ++p)
                {
                    if(cm.getOp(*p)->hasAttr("Workloads0"))
                        workload += cm.getOp(*p)->get<mv::Workloads>("Workloads0").nWorkloads();
                    if(cm.getOp(*p)->hasAttr("Workloads1"))
                        workload += cm.getOp(*p)->get<mv::Workloads>("Workloads1").nWorkloads();
                    if(cm.getOp(*p)->hasAttr("Workloads2"))
                        workload += cm.getOp(*p)->get<mv::Workloads>("Workloads2").nWorkloads();
                    if(cm.getOp(*p)->hasAttr("Workloads3"))
                        workload += cm.getOp(*p)->get<mv::Workloads>("Workloads3").nWorkloads();
                }

                if(workload >= 128)
                {
                    std::cout << "workload fail" << std::endl;
                    foundSafetyBarrier = false;
                    continue;
                }

                if(!checkCommonConsumer(cm, barrierTasks[oldVirtualID], barrierTasks[srcUpdateBarrier]))
                {
                    // std::cout << "checkCommonConsumer fail" << std::endl;
                    continue;
                }
                if(!checkCircleWithNewBarrier(cm, barrierTasks[srcUpdateBarrier], barrier))
                {
                    // std::cout << "checkCircleWithNewBarrier fail" << std::endl;
                    continue;
                }
                if(!checkTimeConstraint(cm, barrierTasks, srcUpdateBarrier, virtualID))
                {
                    // std::cout << "checkTimeConstraint fail" << std::endl;
                    continue;
                }

                // std::cout << "pass all check" << std::endl;

                // count edges added for each valid safety barrier
                unsigned addControl = 0;
                vector<mv::Control::OpListIterator> targetOps;
                // for newBarrier's producers
                if(foundSafetyBarrier)
                {
                    addControl+= countNewEdge(cm, barrierTasks[srcUpdateBarrier], barrierOp, false, 
                                            foundSafetyBarrier, oldVirtualID, targetOps,
                                            sortedDMAOps, sortedUPAOps, sortedDPUOps, barrierTasks, splitCheck);
                }
                // for newBarrier's consumers
                if(foundSafetyBarrier)
                {
                    addControl+= countNewEdge(cm, barrierTasks[srcUpdateBarrier], barrierOp, true, 
                                            foundSafetyBarrier, oldVirtualID, targetOps, 
                                            sortedDMAOps, sortedUPAOps, sortedDPUOps, barrierTasks, splitCheck);
                }
                else{
                    std::cout << "countNewEdge fail" << std::endl;
                }

                // B0 doesn't have requirement on splitBarrierDependency
                if(referenceDevice == "B0")
                {
                    splitCheck = true;
                }

                auto safeBarrierConsumers = barrierTasks[srcUpdateBarrier]->get<mv::Barrier>("Barrier").getConsumers();

                // avoid to add circular dependency
                if(foundSafetyBarrier && addControl && (!splitCheck))
                {
                    addControl = 1;
                    // int maxSchedulingNumber = 0;
                    std::string finalConsumer;
                    // for(auto& p: safeBarrierConsumers)
                    // {
                    //     auto schedulingNumber = cm.getOp(p)->get<unsigned>("scheduleID");
                    //     if(schedulingNumber > maxSchedulingNumber)
                    //     {
                    //         maxSchedulingNumber = schedulingNumber;
                    //         finalConsumer = p;
                    //     }
                    // }
                    
                    int waitCount = 0;
                    for(auto iter= targetOps.begin(); iter!= targetOps.end(); iter++)
                    {
                        auto barrierDpes = (*iter)->get<mv::BarrierDependencies>("BarrierDeps");
                        waitCount += barrierDpes.getWaitSize();
                    }

                    if(!waitCount)
                    {
                        // std::cout << "safety b " << srcUpdateBarrier << " waitCount is 0" << std::endl;
                        // foundSafetyBarrier = false;
                        for(auto iter= targetOps.begin(); iter!= targetOps.end(); iter++)
                        {
                            if((*iter)->getOpType() != "DMATask")
                            {
                                std::cout << "DPU/UPA Task " << (*iter)->getName() << " doesn't have wait barrier!" << std::endl;
                                exit(0);
                            }    
                        }
                    }

                    for(auto& p: safeBarrierConsumers)
                    {
                        auto baseOp = targetOps[0];                        
                        for (auto iter= targetOps.begin(); iter!= targetOps.end(); iter++)
                        {
                            if((*iter)->hasAttr("addNewBarrier")) //TODO: enable it
                            {
                                // std::cout << "safety b " << srcUpdateBarrier << " add barrier to same task" << std::endl;
                                foundSafetyBarrier = false;
                            }
                            if(cm.pathExists(*iter, cm.switchContext(cm.getOp(p))))
                            {
                                // std::cout << "backward depends fail" << std::endl;
                                foundSafetyBarrier = false;
                            }
                            if((cm.getOp(p)->get<unsigned>("scheduleID") >= (*iter)->get<unsigned>("scheduleID")))
                            {
                                // std::cout << "scheduleID order fail" << std::endl;
                                foundSafetyBarrier = false;
                            }
                            if(!opTypeCheck(om.switchContext(baseOp), om.switchContext(*iter)))
                            {
                                addDifferentTypeOfTask = true;
                                // foundSafetyBarrier = false;
                            }
                            // else
                            // {
                            //     if(baseOp->getOpType() != "DMATask")
                            //         addDifferentTypeOfTask = true;
                            // }                            
                        }
                    }
                }

                if((!addControl) && foundSafetyBarrier)
                {
                    zeroDepPool.push_back(srcUpdateBarrier);              
                    break;
                }
                else if(splitCheck && foundSafetyBarrier)
                {
                    if(addControl < minAddSplit)
                    {
                        splitBarrierPool.clear();
                        splitBarrierPool.push_back(srcUpdateBarrier);
                        minAddSplit = addControl;
                    }
                    else if(addControl == minAddSplit)
                    {
                        splitBarrierPool.push_back(srcUpdateBarrier);
                    }
                }
                else if((!addDifferentTypeOfTask) && foundSafetyBarrier)
                {
                    if(addControl < minAddSplitDPU)
                    {
                        addSameTypeTaskPool.clear();
                        addSameTypeTaskPool.push_back(srcUpdateBarrier);
                        minAddSplitDPU = addControl;
                    }
                    else if(addControl == minAddSplitDPU)
                    {
                        addSameTypeTaskPool.push_back(srcUpdateBarrier);
                    }
                }
                else if(foundSafetyBarrier)
                {
                    if(addControl < minAdd)
                    {
                        filterdIDPool.clear();
                        filterdIDPool.push_back(srcUpdateBarrier);
                        minAdd = addControl;
                    }
                    else if(addControl == minAdd)
                    {
                        filterdIDPool.push_back(srcUpdateBarrier);
                    }
                }

                if(foundSafetyBarrier)
                {
                    res_targetOps.insert(std::pair<unsigned, vector<mv::Control::OpListIterator>>(srcUpdateBarrier, targetOps));
                }
            }
            
            // select safety barrier
            if(zeroDepPool.size())
            {
                filterdID = zeroDepPool[zeroDepPool.size() - 1];
            }
            else if(splitBarrierPool.size())
                filterdID = splitBarrierPool[0];
            else if(addSameTypeTaskPool.size())
                filterdID = addSameTypeTaskPool[0];
            else if(filterdIDPool.size())
                filterdID = filterdIDPool[0];

            if(filterdID < 0)
            {
                // if((!specialUPABarrier) || (physicalID != (physicalBarrierBound - 1)))
                //     needUpdate = true;
            }
            else
            {
                // std::cout << "valid VID: " << filterdID << std::endl;            
                if(res_targetOps[filterdID].size())
                {
                    count++;
                }

                // add new dependency/barrier
                if((zeroDepPool.size() == 0 && splitBarrierPool.size() == 0))
                {
                    // std::cout << "add new barrier" << std::endl; 
                    int maxWaitID = -1;
                    auto newBarrierName = addNewBarrier(cm, barrierTasks[filterdID], res_targetOps[filterdID], barrierTasksNum, maxWaitID, sortedDMAOps);
                    barrierTasksNum++;
                    newBarrierTasks.push_back(cm.getOp(newBarrierName));
                    if(maxWaitID < 0)
                    {
                        newBarrierTasksID.push_back(virtualID);
                        cm.getOp(newBarrierName)->set<unsigned>("maxWaitID", virtualID);
                    }
                    else
                    {
                        newBarrierTasksID.push_back(maxWaitID);
                        cm.getOp(newBarrierName)->set<unsigned>("maxWaitID", maxWaitID);
                    }
                    barrierTasks.push_back(cm.getOp(newBarrierName));     
                }
                else if(zeroDepPool.size() == 0)
                {
                    // std::cout << "add new edge" << std::endl; 
                    addNewEdge(cm, barrierTasks[filterdID], res_targetOps[filterdID]);
                }

                // add safety barrier information
                if(barrierTasks[filterdID]->hasAttr("safetyBarrierCount"))
                {
                    auto& safetyBarrierCount = barrierTasks[filterdID]->get<unsigned>("safetyBarrierCount");
                    safetyBarrierCount++;
                }
                else
                {
                    barrierTasks[filterdID]->set<unsigned>("safetyBarrierCount",1);
                }
            
                barrierOp->set<std::string>("safeBarrierName", barrierTasks[filterdID]->getName());                    
                auto oldReadyForConsume = barrierTasks[filterdID]->hasAttr("readyForConsume") ? barrierTasks[filterdID]->get<unsigned>("readyForConsume") : 0;
                oldReadyForConsume = max(oldReadyForConsume, barrierOp->get<unsigned>("scheduleID"));
                barrierTasks[filterdID]->set<unsigned>("readyForConsume", oldReadyForConsume);
                std::cout << "old virtual is " << oldVirtualID << ", safety barrier of " << virtualID << " is " << filterdID << std::endl;
            }
        }

        toVirtual[physicalID] = virtualID;
        // newBarrierTasks.push_back(barrierOp);
    }

    std::cout << "sort newBarrierTasks" << std::endl;
    std::sort(newBarrierTasksID.begin(), newBarrierTasksID.end());
    std::sort(newBarrierTasks.begin(), newBarrierTasks.end(), 
    [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<unsigned>("maxWaitID") < b->get<unsigned>("maxWaitID"); });

    unsigned shift = 0;
    unsigned start = 0;
    for(unsigned i = 0; i < newBarrierTasksID.size(); i++)
    {
        auto end = newBarrierTasksID[i];
        std::cout << "insert " << shift << " " << start << " " << end << " " << std::endl;
        newBarrierTasks.insert(newBarrierTasks.begin() + shift, barrierTasks.begin() + start, barrierTasks.begin() + end + 1);
        shift += (end - start + 2);
        start = end + 1;
    }

    newBarrierTasks.insert(newBarrierTasks.begin() + shift, barrierTasks.begin() + start, barrierTasks.end() - newBarrierTasksID.size());

    std::cout << "insert done" << std::endl;

    // for(int i = 0; i < newBarrierTasks.size(); i++)
    // {
    //     std::cout << newBarrierTasks[i]->getName() << std::endl;;
    // }

    // assign new barrier virtual ID
    int newID = 0;
    for(unsigned i = 0; i < newBarrierTasks.size(); i++)
    {
        mv::Barrier &barrier = newBarrierTasks[i]->get<mv::Barrier>("Barrier");
        barrier.setID(newID);
        barrier.setIndex(newID);
        newID++;
    }

    updateBarrierDependency(cm);

    for(unsigned i = 0; i < newBarrierTasks.size(); i++)
    {
        if(newBarrierTasks[i]->hasAttr("safeBarrierName"))
        {
            auto name = newBarrierTasks[i]->get<std::string>("safeBarrierName");
            mv::Barrier &barrier = cm.getOp(name)->get<mv::Barrier>("Barrier");
            mv::BarrierDependencies addSafetyBarrier;
            addSafetyBarrier.addUpdateBarrier(barrier.getID());
            newBarrierTasks[i]->set<mv::BarrierDependencies>("BarrierDeps", addSafetyBarrier);
        }
    }

    return (count > 0);
}

void RemoveRedundantBarriersForDMA(mv::ComputationModel& model){
  
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    auto bOps = om.getOps("BarrierTask");
    auto dmaOps = cm.schedulingSortDMA();
    std::sort(
            bOps.begin(),
            bOps.end(),
            [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
            );
    unsigned int removed_updates = 0, removed_waits = 0;
    
    for (unsigned i=0; i< dmaOps.size(); i++)
    {
        auto& opIt= dmaOps[i];
        if (!(opIt->hasAttr("BarrierDeps"))) { continue; }
        auto barrierDeps = opIt->get<mv::BarrierDependencies>("BarrierDeps");
        const std::vector<unsigned>& waits = barrierDeps.getWait();
        for(auto& iwb: waits){
            for(unsigned j=i+1; j< dmaOps.size(); j++){
                auto& barrierDeps_j = dmaOps[j]->get<mv::BarrierDependencies>("BarrierDeps");
                const std::vector<unsigned>& waits_j = barrierDeps_j.getWait();
                for(unsigned jbi=0; jbi< waits_j.size(); jbi++){
                    auto jwb= waits_j[jbi];
                    if(iwb== jwb){
                    mv::Logger::log(mv::Logger::MessageType::Debug, "BarrierSplit", 
                        "Removing wait for V: "+ std::to_string(jwb) +", from DMA task "+std::to_string(j));
                    
                    mv::Barrier &barrier = bOps[jwb]->get<mv::Barrier>("Barrier");
                    barrier.removeConsumer(dmaOps[j]->getName());
                    
                    barrierDeps_j.removeWaitBarrier(jwb);

                    auto sourceFlowStart = cm.switchContext(bOps[jwb]).leftmostOutput();
                    for (mv::Control::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != cm.flowEnd(); ++sinkFlow)
                    {
                        if(sinkFlow.sink()->getName() == dmaOps[j]->getName())
                            cm.undefineFlow(sinkFlow);
                    }
                    
                    jbi--;
                    removed_waits++;
                    }
                }
            }
        }
    }
    
    for (int i=dmaOps.size()-1; i>= 0; i--)
    {
        auto& opIt= dmaOps[i];
        if (!(opIt->hasAttr("BarrierDeps"))) { continue; }
        auto barrierDeps = opIt->get<mv::BarrierDependencies>("BarrierDeps");
        const std::vector<unsigned>& updates = barrierDeps.getUpdate();
        for(auto& iub: updates){
            for(int j=i-1; j>= 0; --j){
            auto& barrierDeps_j = dmaOps[j]->get<mv::BarrierDependencies>("BarrierDeps");
            const std::vector<unsigned>& updates_j = barrierDeps_j.getUpdate();
                for(unsigned jbi=0; jbi< updates_j.size(); jbi++){
                    auto jub= updates_j[jbi];
                    if(iub== jub){
                    mv::Logger::log(mv::Logger::MessageType::Debug, "BarrierSplit", 
                        "Removing wait for V: "+ std::to_string(jub) +", from DMA task "+std::to_string(j));
                    mv::Barrier &barrier = bOps[jub]->get<mv::Barrier>("Barrier");
                    barrier.removeProducer(dmaOps[j]->getName());
                    
                    barrierDeps_j.removeUpdateBarrier(jub);

                    auto sourceFlowStart = (dmaOps[j]).leftmostOutput();
                    for (mv::Control::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != cm.flowEnd(); ++sinkFlow)
                    {
                        if(sinkFlow.sink()->getName() == bOps[jub]->getName())
                            cm.undefineFlow(sinkFlow);
                    }
                    
                    jbi--;
                    removed_updates++;
                    }
                }
            }
        }
    }
    
    mv::Logger::log(mv::Logger::MessageType::Debug, "SplitBarrier", "Removed "+ std::to_string(removed_waits)+ " wait and "+ std::to_string(removed_updates) +" update barrier counts for DMA tasks");
}

int markUPAWaitBarrier(mv::ControlModel& cm)
{
    auto sortedOps = cm.schedulingSortDPUorUPA();
    vector<mv::Control::OpListIterator>  upaTasks;
    for(unsigned i = 0; i < (sortedOps.size()); i++)
    {
        if(sortedOps[i]->getOpType() == "UPATask")
        {
            upaTasks.push_back(sortedOps[i]);
        }        
    }

    auto bTasks = cm.getOps("BarrierTask");
    std::sort(
    bTasks.begin(),
    bTasks.end(),
    [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
    );

    for(auto& op: bTasks)
    {
        if(op->hasAttr("upaConsumer"))
        {
            op->erase("upaConsumer");
        }
    }

    int specialUPABarrier = 0;

    for(auto& op: upaTasks)
    {
        auto barrierDeps = op->get<mv::BarrierDependencies>("BarrierDeps");
        auto waitList = barrierDeps.getWait();
        auto upaConsumer = true;

        for(auto& wid: waitList)
        {
            auto prodName = bTasks[wid]->get<mv::Barrier>("Barrier").getProducers();
            for(auto& name: prodName)
            {
                if(cm.getOp(name)->getOpType() == "UPATask")
                {
                    upaConsumer = false;
                }
            }
        }

        if(upaConsumer)
        {
            bool findWaitBarrier = false;
            for(auto& wid: waitList)
            {
                bool allUPA = true;
                auto consumerName = bTasks[wid]->get<mv::Barrier>("Barrier").getConsumers();
                for(auto& name: consumerName)
                {
                    if(cm.getOp(name)->getOpType() != "UPATask")
                    {
                        allUPA = false;
                        break;
                    }
                }
                if(allUPA)
                {
                    if(!specialUPABarrier)
                        bTasks[wid]->set<bool>("specialUPABarrier", true);
                    if(!bTasks[wid]->hasAttr("upaConsumer"))
                    {
                        bTasks[wid]->set<unsigned>("upaConsumer", specialUPABarrier);
                        specialUPABarrier++;
                    }
                    findWaitBarrier = true;
                    break;
                }
            }

            if(!findWaitBarrier)
            {
                if(waitList.size())
                {
                    if(!specialUPABarrier)
                        bTasks[waitList[0]]->set<bool>("specialUPABarrier", true);
                    if(!bTasks[waitList[0]]->hasAttr("upaConsumer"))
                    {
                        bTasks[waitList[0]]->set<unsigned>("upaConsumer", specialUPABarrier);
                        specialUPABarrier++;
                    }
                }
            }
        }
    }

    return specialUPABarrier;
}

void splitBarrierForDPUAndUPA(mv::ComputationModel& model, string baseName)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    auto bOps = om.getOps("BarrierTask");

    std::sort(
            bOps.begin(),
            bOps.end(),
            [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
            );

    std::vector<mv::Data::OpListIterator> newBarrierTasks;
    auto barrier_task_id = bOps.size();

    for(auto& bOp : bOps)
    {
        auto& barrier = bOp->get<mv::Barrier>("Barrier");
        auto consumers = barrier.getConsumers();

        bool hasDPUConsumer = false;
        bool hasUPAConsumer = false;
        std::vector<std::string> UPATaskNames;

        for(auto p = consumers.begin(); p != consumers.end(); ++p)
        {
            if(om.getOp(*p)->getOpType() == "DPUTask")
                hasDPUConsumer = true;
            if(om.getOp(*p)->getOpType() == "UPATask")
            {
                hasUPAConsumer = true;
                UPATaskNames.push_back(*p);
            }
        }

        if(hasDPUConsumer && hasUPAConsumer)
        {
            char barrier_name[64UL]; 
            sprintf(barrier_name, "UPA_Barrier_%zu", barrier_task_id);

            std::set<std::string> empty_set;
            struct mv::Barrier new_barrier(empty_set, empty_set);

            om.barrierTask(baseName + barrier_name, new_barrier);
            auto barrier_new = om.getOp(baseName + barrier_name);
            assert((barrier_new != om.opEnd()) &&
                    barrier_new->hasAttr("Barrier"));
            mv::Barrier &barrier_barrier_new = barrier_new->get<mv::Barrier>("Barrier");
            barrier_barrier_new.setID(barrier_task_id);
            barrier_barrier_new.setIndex(barrier_barrier_new.getID());
            barrier_barrier_new.setRealBarrierIndex(-1);
            barrier_task_id++;

            auto producers = barrier.getProducers();
            for(auto prod = producers.begin(); prod != producers.end(); ++prod)
            {
                barrier_barrier_new.addProducer(*prod);
                mv::BarrierDependencies& barrierDeps = om.getOp(*prod)->get<mv::BarrierDependencies>("BarrierDeps");
                barrierDeps.addUpdateBarrier(barrier_barrier_new.getIndex());

                cm.defineFlow(om.getOp(*prod), barrier_new);
            }

            for(auto upa = UPATaskNames.begin(); upa != UPATaskNames.end(); ++upa)
            {
                barrier.removeConsumer(*upa);

                mv::BarrierDependencies& barrierDeps = om.getOp(*upa)->get<mv::BarrierDependencies>("BarrierDeps");
                barrierDeps.removeWaitBarrier(barrier.getIndex());

                auto sourceFlowStart = cm.switchContext(bOp).leftmostOutput();
                for (mv::Control::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != cm.flowEnd(); ++sinkFlow)
                {
                    if(sinkFlow.sink()->getName() == *upa)
                        cm.undefineFlow(sinkFlow);
                }

                barrier_barrier_new.addConsumer(*upa);
                barrierDeps.addWaitBarrier(barrier_barrier_new.getIndex());
                cm.defineFlow(barrier_new, om.getOp(*upa));                
            }
            
            newBarrierTasks.push_back(bOp);
            newBarrierTasks.push_back(barrier_new);
        }
        else
        {
            newBarrierTasks.push_back(bOp);
        }        
    }

    // assign new barrier virtual ID
    int newID = 0;
    for(unsigned i = 0; i < newBarrierTasks.size(); i++)
    {
        mv::Barrier &barrier = newBarrierTasks[i]->get<mv::Barrier>("Barrier");
        barrier.setID(newID);
        barrier.setIndex(newID);
        newID++;
    }

    updateBarrierDependency(cm);
}

// For details please refer to the document on this link https://intel-my.sharepoint.com/:w:/p/shaojun_yao/EcF9B7gifspDvWIyBLEyK54BEL0hAooR0R9rfvWDKpDdHw?e=8nFb8o
static void AssignSafetyBarrierFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    bool assignSafetyBarrier = false;
    auto globalParams = model.getGlobalConfigParams();

    if(globalParams->hasAttr("enableBarrierSafety"))
    {
      assignSafetyBarrier = globalParams->get<bool>("enableBarrierSafety");
    }

    if(assignSafetyBarrier)
    {
        std::cout << "assign safety barrier" << std::endl;
        std::string referenceDevice = "A0";
        if(globalParams->hasAttr("referenceDevice"))
        {
            referenceDevice = globalParams->get<std::string>("referenceDevice");
        }
        size_t real_physical_barriers = (size_t) model.getGlobalConfigParam("real_physical_barriers").get<int>();
        if(real_physical_barriers == 0)
            real_physical_barriers = (size_t) passDesc.get<int>("real_physical_barriers");
        size_t count = 0;
        char baseName[64UL];

        mv::OpModel om(model);
        mv::ControlModel cm(model);
        std::vector<mv::Control::OpListIterator> sortedDPUorUPA = cm.schedulingSortDPUorUPA();
        bool success = true;  
        bool status = true;

        while(status)
        {        
            sprintf(baseName, "iter_%zu_", count);
            RemoveRedundantBarriersForDMA(model);
            if(referenceDevice == "A0")
            {
                splitBarrierDependencies(model, baseName);
                if(!splitBarrierCheck(model))
                {
                    pass.log(mv::Logger::MessageType::Error, "splitBarrierCheck fails after splitBarrierDependencies");
                }
            }

            // splitBarrierForDPUAndUPA(model, baseName);

            int specialUPABarrier = markUPAWaitBarrier(cm);
        
            success =
                    mv::lp_scheduler::Control_Model_Barrier_Assigner::assign_physical_id(cm,
                        real_physical_barriers, specialUPABarrier ? 1: 0);

            if(!success)
            {
                pass.log(mv::Logger::MessageType::Error, "assign_physical_id fails");
                exit(0);
            }

            // assign dedicated physical ID for marked UPA wait barrier
            auto bTasks = cm.getOps("BarrierTask");
            std::sort(
            bTasks.begin(),
            bTasks.end(),
            [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
            );

            std::cout << "modify the rest time for upaConsumer" << std::endl;
            int resetID = -1;
            // if(count < 2)
            {
                for(auto& op: bTasks)
                {
                    if(op->hasAttr("upaConsumer"))
                    {
                        std::cout << op->getName() << std::endl;
                        mv::Barrier &barrier = op->get<mv::Barrier>("Barrier");
                        barrier.setRealBarrierIndex(real_physical_barriers - 1);
                        if(resetID < 0)
                        {
                            resetID = op->get<unsigned>("readyForReset");
                        }
                        else
                        {
                            op->set<unsigned>("scheduleID", resetID);
                            resetID = op->get<unsigned>("readyForReset");
                        }                
                    }
                }
            std::cout << "modify the rest time for upaConsumer done" << std::endl;
            }

            // for(auto& op: bTasks)
            // {
            //     auto barrierProduers = op->get<mv::Barrier>("Barrier").getProducers();
            //     auto barrierConsumers = op->get<mv::Barrier>("Barrier").getConsumers();
            //     int maxSchedulingNumber = 0;
            //     std::string finalConsumer;
            //     for(auto& p: barrierProduers)
            //     {
            //         auto schedulingNumber = cm.getOp(p)->get<unsigned>("schedulingNumber");
            //         if(schedulingNumber > maxSchedulingNumber)
            //         {
            //             maxSchedulingNumber = schedulingNumber;
            //             finalConsumer = p;
            //         }
            //     }
            //     op->set<unsigned>("readyForConsume", maxSchedulingNumber);

            //     maxSchedulingNumber = 0;
            //     for(auto& p: barrierConsumers)
            //     {
            //         auto schedulingNumber = cm.getOp(p)->get<unsigned>("schedulingNumber");
            //         if(schedulingNumber > maxSchedulingNumber)
            //         {
            //             maxSchedulingNumber = schedulingNumber;
            //             finalConsumer = p;
            //         }
            //     }
            //     op->set<unsigned>("readyForReset", maxSchedulingNumber);
            // }

            // if(count < 2)
                status = barrierSafety(cm, real_physical_barriers, count, specialUPABarrier, referenceDevice);
            // else
                // status = false;
            count++;        
        }
    }
}