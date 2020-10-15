#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"

class Recompute_Memory_Locations {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef mv::Op const * operation_t;
    typedef mv::Op * operation_non_const_t;
    typedef std::list<operation_t> op_list_t;
    typedef mv::Tensor::MemoryLocation mem_location_t;
    typedef std::unordered_map<operation_t, mem_location_t>
        mem_location_table_t;

    class exception_t : std::string {
      public:
        exception_t(const std::string& msg) : std::string(msg) {}
        exception_t(const char *msg) : std::string(msg) {}
        const std::string& getMessage() const { return  *this; }
    }; // class exception_t //
    ////////////////////////////////////////////////////////////////////////////

    Recompute_Memory_Locations(mv::OpModel& om) : omodel_(om) {}

    mem_location_t get_mem_location_of_real_op(operation_t op_in) const {
      operation_non_const_t op = const_cast<operation_non_const_t>(op_in);
      if (!op->outputSlots()) { return mem_location_t(); }
      mv::Data::TensorIterator tensor_itr = op->getOutputTensor(0UL);
      return tensor_itr->get<mem_location_t>("Location");
    }

    template<typename T>
    size_t get_in_degree(T op) const {
      size_t in_degree = 0UL;
      for (auto pitr=op.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        ++in_degree;
      }
      return in_degree;
    }

    void dump(FILE *fptr=stdout) const {
      for (auto itr=mem_location_table_.begin(); itr!=mem_location_table_.end();
            ++itr) {
        fprintf(fptr, "op=%s mem_location=%s\n",
              (itr->first)->getName().c_str(),
              (itr->second).toString().c_str());
      }
    }

    void set_recomputed_memory_locations() {
      for (auto itr=mem_location_table_.begin(); itr!=mem_location_table_.end();
            ++itr) {
        mv::Data::OpListIterator op_itr =
            omodel_.getOp((itr->first)->getName());
        mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(0UL);
        tensor_itr->set<mem_location_t>("Location", itr->second);
      }
    }

    size_t recompute() {
      mem_location_table_t &implicit_op_mem_table = mem_location_table_;
      std::unordered_map<operation_t, size_t> in_degree_map;
      op_list_t zero_in_degree_nodes[2UL];
      mv::OpModel &om=omodel_;

      implicit_op_mem_table.clear();

      for (auto itr=om.opBegin(); itr!=om.opEnd(); ++itr) {
        size_t in_degree = get_in_degree(itr);
        operation_t op = &(*itr);

        if (!in_degree) {
          zero_in_degree_nodes[0UL].push_back(op);
          if (op->isImplicit()) {
            throw exception_t("Implicit Ops cannot have zero in degree " +
                  op->getName());
          }
        }
        in_degree_map[op] = in_degree;
      }

      bool parity = false;
      while (!zero_in_degree_nodes[parity].empty()) {
        op_list_t& curr_level = zero_in_degree_nodes[parity];
        op_list_t& next_level = zero_in_degree_nodes[!parity];

        for (auto itr=curr_level.begin(); itr!=curr_level.end(); ++itr) {
          operation_t zop = *itr;
          mem_location_t zop_mem_location;

          // inductive argument: any implicit op should have a valid memory
          // location//
          if (zop->isImplicit()) {
            auto imop = implicit_op_mem_table.find(zop);
            if (imop == implicit_op_mem_table.end())
              throw mv::RuntimeError("LpScheduler", "Inductive invariant violation.");
            zop_mem_location = imop->second;
          } else {
            zop_mem_location = get_mem_location_of_real_op(zop);
          }

          mv::Data::OpListIterator zop_itr = omodel_.getOp(zop->getName());
          for (auto citr=zop_itr.leftmostChild(); citr!=omodel_.opEnd(); ++citr)
          {
            operation_t cop = &(*citr);
            auto ditr = in_degree_map.find(cop);

            if ((ditr == in_degree_map.end()) || (ditr->second == 0)) {
              throw exception_t("in_degree_map invariant violation\n");
            }

            // maintain the inductive invariant //
            if (cop->isImplicit()) {
              // if already has an entry make sure the memory location uniform.
              auto mitr = implicit_op_mem_table.find(cop);
              if (mitr == implicit_op_mem_table.end()) {
                mitr = implicit_op_mem_table.insert(
                    std::make_pair(cop, zop_mem_location)).first;
              } else if (!(mitr->second  == zop_mem_location) ) {
                throw exception_t("Implicit op " + cop->getName() +
                      " has memory location un-resolved");
              }
            }

            --(ditr->second);
            if (!(ditr->second)) {
              next_level.push_back(ditr->first);
            }
          }
        }
        curr_level.clear();
        parity = !parity;
      }

      return implicit_op_mem_table.size();
    }

  private:

    mv::OpModel &omodel_;
    mem_location_table_t mem_location_table_;
}; // class Recompute_Memory_Locations //


static void RecomputeImplicitOpMemoryLocations(const mv::pass::PassEntry&,
    mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv {
namespace pass {

MV_REGISTER_PASS(RecomputeImplicitOpMemoryLocations)
  .setFunc(RecomputeImplicitOpMemoryLocations);

} // namespace pass //
} // namespace mv//

void RecomputeImplicitOpMemoryLocations(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc,
      mv::Element&) {
  mv::OpModel om(model);
  Recompute_Memory_Locations computer(om);
  computer.recompute();
  computer.set_recomputed_memory_locations();

  if (passDesc.hasAttr("output")) {
    std::string output_file = passDesc.get<std::string>("output");
    FILE *fptr = fopen(output_file.c_str(), "w");
    if (fptr != nullptr) {
      computer.dump(fptr);
      fclose(fptr);
    } else {
      throw mv::RuntimeError("LpScheduler", "Can't write to the file " + output_file);
    }
  }
}
