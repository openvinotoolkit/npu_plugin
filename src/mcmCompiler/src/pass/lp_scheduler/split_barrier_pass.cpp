#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "lp_scheduler/barrier_scheduler_pass.hpp"

static void SplitBarrierFcn(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void RemoveRedundantBarriersForDMA(mv::ComputationModel& model);


namespace mv {
  namespace pass {

    MV_REGISTER_PASS(SplitBarrier)
      .setFunc(SplitBarrierFcn)
      .setDescription("Split barriers for A0 workaround and make sure a barrier only has dma consumers or none-dma comsumers");

  } // namespace mv //
} // namespace pass //

static void SplitBarrierFcn(
    const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor& targetDesc, mv::Element&, mv::Element&){
  
  mv::ControlModel cmodel(model);
  mv::OpModel om(model);
  auto bOps = om.getOps("BarrierTask");
  std::vector<std::pair<unsigned short, unsigned short>> dmaContrib(bOps.size());
  auto dmaOps = om.getOps("DMATask");

  bool isStatic= false, isA0= true;
  auto globalParams = model.getGlobalConfigParams();
  if (globalParams->hasAttr("enableStaticBarriers"))
  {
    isStatic= globalParams->get<bool>("enableStaticBarriers");
  }
  if (globalParams->hasAttr("referenceDevice") && globalParams->get<std::string>("referenceDevice") == "B0"){
    isA0= false;
  }
  // THB
  if (targetDesc.getTarget()==mv::Target::ma3100){
    isA0= false;
  }
  if (!(isStatic && isA0))
  {
    mv::Logger::log(mv::Logger::MessageType::Debug, "SplitBarrier", "No SplitBarrier WA");
    return;
  }
  
  mv::Logger::log(mv::Logger::MessageType::Debug, "SplitBarrier", "Static barriers assignment with SplitBarrier WA for A0");

  // NN_REMOVE_REDUNDANT_BARRIERS_1808882661 workaround
  RemoveRedundantBarriersForDMA(model);
  
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
  auto barrier_task_id = 0;
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
          sprintf(barrier_name, "new_Barrier_%d", barrier_task_id);

          std::set<std::string> empty_set;
          struct mv::Barrier new_barrier(empty_set, empty_set);

          om.barrierTask(barrier_name, new_barrier);
          auto barrier_new = om.getOp(barrier_name);
          assert((barrier_new != om.opEnd()) &&
                  barrier_new->hasAttr("Barrier"));
          mv::Barrier &barrier_barrier_new = barrier_new->get<mv::Barrier>("Barrier");
          barrier_barrier_new.setID(barrier_task_id);
          barrier_barrier_new.setIndex(barrier_barrier_new.getID());

          mappings[i].first = static_cast<short>(barrier_task_id);
          barrier_task_id++;
          newBarrierNames.push_back(barrier_name);
      }

      if (gamma > 0)
      {
        // create new barriers
          char barrier_name[64UL];
          sprintf(barrier_name, "new_Barrier_%d", barrier_task_id);

          std::set<std::string> empty_set;
          struct mv::Barrier new_barrier(empty_set, empty_set);

          om.barrierTask(barrier_name, new_barrier);
          auto barrier_new = om.getOp(barrier_name);
          assert((barrier_new != om.opEnd()) &&
                  barrier_new->hasAttr("Barrier"));
          mv::Barrier &barrier_barrier_new = barrier_new->get<mv::Barrier>("Barrier");
          barrier_barrier_new.setID(barrier_task_id);
          barrier_barrier_new.setIndex(barrier_barrier_new.getID());

          mappings[i].second = static_cast<short>(barrier_task_id);
          barrier_task_id++;
          newBarrierNames.push_back(barrier_name);
      }
      
      mv::Logger::log(mv::Logger::MessageType::Debug, "SplitBarrier", "Barrier "
                      + std::to_string(i)+ ": alpha: "+ std::to_string(alpha)+ ", beta: "+ std::to_string(beta)
                      + ", delta: "+ std::to_string(delta)+ ", gamma: "+ std::to_string(gamma)
                      + ", mapping: "+ std::to_string(mappings[i].first)+ " - " + std::to_string(mappings[i].second));
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

static void RemoveRedundantBarriersForDMA(mv::ComputationModel& model){
  
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
  
  for (int i=0; i< dmaOps.size(); i++)
  {
      auto& opIt= dmaOps[i];
      if (!(opIt->hasAttr("BarrierDeps"))) { continue; }
      auto barrierDeps = opIt->get<mv::BarrierDependencies>("BarrierDeps");
      const std::vector<unsigned>& waits = barrierDeps.getWait();
      for(auto& iwb: waits){
        for(int j=i+1; j< dmaOps.size(); j++){
          auto& barrierDeps_j = dmaOps[j]->get<mv::BarrierDependencies>("BarrierDeps");
          const std::vector<unsigned>& waits_j = barrierDeps_j.getWait();
          for(int jbi=0; jbi< waits_j.size(); jbi++){
            auto jwb= waits_j[jbi];
            if(iwb== jwb){
              mv::Logger::log(mv::Logger::MessageType::Debug, "SplitBarrier", 
                  "Removing wait for V: "+ std::to_string(jwb) +", from DMA task "+std::to_string(j));
              
              mv::Barrier &barrier = bOps[jwb]->get<mv::Barrier>("Barrier");
              barrier.removeConsumer(dmaOps[j]->getName());
              
              barrierDeps_j.removeWaitBarrier(jwb);
              
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
          for(int jbi=0; jbi< updates_j.size(); jbi++){
            auto jub= updates_j[jbi];
            if(iub== jub){
              mv::Logger::log(mv::Logger::MessageType::Debug, "SplitBarrier", 
                  "Removing wait for V: "+ std::to_string(jub) +", from DMA task "+std::to_string(j));
              mv::Barrier &barrier = bOps[jub]->get<mv::Barrier>("Barrier");
              barrier.removeProducer(dmaOps[j]->getName());
              
              barrierDeps_j.removeUpdateBarrier(jub);
              
              jbi--;
              removed_updates++;
            }
          }
        }
      }
  }
  
  mv::Logger::log(mv::Logger::MessageType::Debug, "SplitBarrier", "Removed "+ std::to_string(removed_waits)+ " wait and "+ std::to_string(removed_updates) +" update barrier counts for DMA tasks");
}
