#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"

namespace {

typedef mv::Tensor::MemoryLocation mem_location_t;
typedef mv::Op const * operation_t;
typedef mv::Op * operation_non_const_t;

}

template<typename T>
class Attrs {
  public:
    typedef std::unordered_map<operation_t, T> table_t;
    
    Attrs(std::string name) : attr_name_(name) { attr_table_.clear(); }
    virtual ~Attrs() {};

    void propagate_attr_to_child(operation_t child_op, T const& parent_attr) {
      // maintain the inductive invariant //
      if (child_op->isImplicit()) {
        propagate_attr_to_child_impl(child_op, parent_attr);
      }
    }

    T get_attr_of_real_op(operation_t op_in) const {
      operation_non_const_t op = const_cast<operation_non_const_t>(op_in);
      if (!op->outputSlots()) { return T(); }
      mv::Data::TensorIterator tensor_itr = op->getOutputTensor(mv::IO_TENSOR_OUTPUT);
      return tensor_itr->get<T>(attr_name_);
    }

    T get_attr(operation_t op) const {
      // inductive argument: any implicit op should have a valid attr
      if (op->isImplicit() &&
          (attr_table_.find(op) == attr_table_.end()) ) {
        throw mv::RuntimeError("LpScheduler", "Inductive invariant violation.");
      }

      return op->isImplicit() ?
        (attr_table_.find(op))->second :
        get_attr_of_real_op(op);
    }

    void set_recomputed_attr(mv::OpModel &omodel_) {
      for (const auto& itr : attr_table_) {
        mv::Data::OpListIterator op_itr =
            omodel_.getOp((itr.first)->getName());
        mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        tensor_itr->set<T>(attr_name_, itr.second);
      }
    }

    void dump(FILE *fptr = stdout) const {
      std::string message = "op=%s " + attr_name_ + "=%s\n";
      for (const auto& itr : attr_table_) {
        fprintf(fptr, message.c_str(),
              (itr.first)->getName().c_str(),
              attr_val_to_string(itr.second).c_str());
      }
    }

    std::size_t table_size() const {
      return attr_table_.size();
    }

  protected:
    std::string attr_name_;
    table_t attr_table_;

    virtual std::string attr_val_to_string(T const&) const = 0;
    virtual void propagate_attr_to_child_impl(operation_t, T const&) = 0;
}; // class Attrs<T> //

class Mem_Loc_Attr final: public Attrs<mem_location_t> {
  public:
    Mem_Loc_Attr() : Attrs<mem_location_t>("Location") {}

  private:
    void propagate_attr_to_child_impl(operation_t child_op, mem_location_t const& parent_mem_loc) override {
      // if already has an entry make sure the memory location uniform.
      auto mitr = attr_table_.find(child_op);
      if (mitr == attr_table_.end()) {
        mitr = attr_table_.insert(
            std::make_pair(child_op, parent_mem_loc)).first;
      } else if (!(mitr->second  == parent_mem_loc) ) {
        throw mv::RuntimeError("LpScheduler", "Implicit op " + child_op->getName() +
              " has memory location un-resolved");
      }
    }

    std::string attr_val_to_string(mem_location_t const& val) const override {
      return val.toString();
    }
}; // class Mem_Loc_Attr //

class Recompute_Attrs {
  public:
    ////////////////////////////////////////////////////////////////////////////

    typedef std::list<operation_t> op_list_t;

    ////////////////////////////////////////////////////////////////////////////

    Recompute_Attrs(mv::OpModel& om) : omodel_(om) {
      compute_ops_in_degree();
    }

    template<typename T>
    size_t recompute(Attrs<T> &attr) {
      op_list_t zid_nodes[2UL];
      zid_nodes[0UL] = zero_in_degree_nodes_;

      bool parity = false;
      while (!zid_nodes[parity].empty()) {
        op_list_t& curr_level = zid_nodes[parity];
        op_list_t& next_level = zid_nodes[!parity];

        for (const auto& zop : curr_level) {
          auto zop_attr = attr.get_attr(zop);

          mv::Data::OpListIterator zop_itr = omodel_.getOp(zop->getName());
          for (auto citr = zop_itr.leftmostChild(); citr != omodel_.opEnd(); ++citr)
          {
            operation_t cop = &(*citr);
            auto ditr = in_degree_map_.find(cop);

            if ((ditr == in_degree_map_.end()) || (ditr->second == 0)) {
              throw mv::RuntimeError("LpScheduler", "in_degree_map invariant violation\n");
            }

            attr.propagate_attr_to_child(cop, zop_attr);
            --(ditr->second);
            if (!(ditr->second)) {
              next_level.push_back(ditr->first);
            }
          }
        }
        curr_level.clear();
        parity = !parity;
      }

      return attr.table_size();
    }

  private:
    mv::OpModel &omodel_;
    std::unordered_map<operation_t, size_t> in_degree_map_;
    op_list_t zero_in_degree_nodes_;

    size_t get_in_degree(mv::Data::OpListIterator op) const {
      size_t in_degree = 0UL;
      for (auto pitr = op.leftmostParent(); pitr != omodel_.opEnd(); ++pitr) {
        ++in_degree;
      }
      return in_degree;
    }

    void compute_ops_in_degree() {
      mv::OpModel &om=omodel_;
      for (auto itr = om.opBegin(); itr != om.opEnd(); ++itr) {
        size_t in_degree = get_in_degree(itr);
        operation_t op = &(*itr);

        if (!in_degree) {
          zero_in_degree_nodes_.push_back(op);
          if (op->isImplicit()) {
            throw mv::RuntimeError("LpScheduler", "Implicit Ops cannot have zero in degree " +
                  op->getName());
          }
        }
        in_degree_map_[op] = in_degree;
      }
    }
}; // class Recompute_Attrs //


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
  Recompute_Attrs computer(om);

  Mem_Loc_Attr memLocationAttribute;
  computer.recompute<mem_location_t>(memLocationAttribute);
  memLocationAttribute.set_recomputed_attr(om);

  if (passDesc.hasAttr("output")) {
    std::string output_file = passDesc.get<std::string>("output");
    FILE *fptr = fopen(output_file.c_str(), "w");
    if(fptr == nullptr) {
      throw mv::RuntimeError("LpScheduler", "Can't open file " + output_file);
    }
    memLocationAttribute.dump(fptr);
    fclose(fptr);
  }
}
