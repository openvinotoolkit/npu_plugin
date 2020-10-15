#ifndef REMOVE_REDUNDANT_UPDATE_BARRIERS_HPP
#define REMOVE_REDUNDANT_UPDATE_BARRIERS_HPP

#include <sstream>

#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/op_model.hpp"

namespace mv {
namespace lp_scheduler {


class Remove_Redundant_Update_Barriers {
  public:
    Remove_Redundant_Update_Barriers(mv::ControlModel& cmodel)
      : cmodel_(cmodel) {}


    size_t remove() {

      size_t updated = 0UL;
      for (auto oitr=cmodel_.getFirst(); oitr!=cmodel_.opEnd(); ++oitr) {
        if (!(oitr->hasAttr("BarrierDeps"))) { continue; }

        mv::BarrierDependencies new_deps,
          old_deps = oitr->get<mv::BarrierDependencies>("BarrierDeps");
        std::vector<std::string> new_producer_tasks;

        const std::vector<unsigned>& old_wait = old_deps.getWait();
        for (auto witr=old_wait.begin(); witr!=old_wait.end(); ++witr) {
          new_deps.addWaitBarrier(*witr);
        }

        for (auto citr=oitr.leftmostChild(); citr!=cmodel_.opEnd(); ++citr) {
          if ( !(citr->getOpType() == "BarrierTask") ) { continue; }

          int bindex = (citr->get<mv::Barrier>("Barrier")).getIndex();
          assert(bindex >= 0);
          new_deps.addUpdateBarrier(unsigned(bindex));
          new_producer_tasks.push_back(citr->getName());
        }

        if (new_deps.getUpdate() != old_deps.getUpdate()) {
          log_barrier_update(oitr->getName(),
              old_deps.getUpdate(), new_deps.getUpdate());

          oitr->set<mv::BarrierDependencies>("BarrierDeps", new_deps);
          oitr->set<std::vector<std::string>>("BarriersProducedByTask",
                new_producer_tasks);
          ++updated;
        }
      }

      return updated;
    }

  private:

    template<typename BarrierIdContainer>
    void log_barrier_update(const std::string& op_name,
        const BarrierIdContainer& old_ids,
        const BarrierIdContainer& new_ids) const {

      std::ostringstream log_stream;
      log_stream << "[UpdateBarrierDeps]: op=" << op_name; 

      {
        log_stream << " old_update_deps=[ "; 
        for (auto bitr=old_ids.begin(); bitr!=old_ids.end(); ++bitr) {
          log_stream << *bitr << " ";
        }
        log_stream << " ] ";
      }

      {
        log_stream << " new_update_deps=[ "; 
        for (auto bitr=new_ids.begin(); bitr!=new_ids.end(); ++bitr) {
          log_stream << *bitr << " ";
        }
        log_stream << " ] ";
      }
      log_stream << std::endl;

      mv::Logger::log(mv::Logger::MessageType::Info,
          "RemoveRedundantUpdateBarriers", log_stream.str());
    }


    mv::ControlModel& cmodel_;
}; // class Remove_Redundant_Update_Barriers //

} // namespace lp_scheduler //
} // namespace mv //

#endif
