#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"

namespace {

typedef mv::Tensor::MemoryLocation mem_location_t;
typedef mv::Op const * operation_t;
typedef mv::Op * operation_non_const_t;
typedef std::list<operation_t> op_list_t;
typedef std::unordered_map<operation_t, size_t> degree_map_t;

size_t get_in_degree(mv::Data::OpListIterator op, mv::OpModel &om) {
  size_t in_degree = 0UL;
  for (auto pitr = op.leftmostParent(); pitr != om.opEnd(); ++pitr) {
    ++in_degree;
  }
  return in_degree;
}

void compute_ops_in_degree(mv::OpModel &om, op_list_t& zero_in_degree_nodes, degree_map_t& in_degree_map) {
  for (auto itr = om.opBegin(); itr != om.opEnd(); ++itr) {
    size_t in_degree = get_in_degree(itr, om);
    operation_t op = &(*itr);

    if (!in_degree) {
      zero_in_degree_nodes.push_back(op);
      if (op->isImplicit()) {
        throw mv::RuntimeError("LpScheduler", 
            "Implicit Ops cannot have zero in degree " + op->getName());
      }
    }
    in_degree_map[op] = in_degree;
  }
}

}

template<typename T>
class Attrs {
  public:
    typedef std::unordered_map<operation_t, T> table_t;
    
    Attrs(std::string name, mv::OpModel& om, op_list_t& zero_in_degree, degree_map_t in_degree_map) :
      attr_name_(name),
      omodel_(om),
      zero_in_degree_nodes_(zero_in_degree),
      in_degree_map_(in_degree_map) {
      attr_table_.clear();
    }

    virtual ~Attrs() {};

    void set_recomputed_attr() {
      for (const auto& itr : attr_table_) {
        mv::Data::OpListIterator op_itr =
            omodel_.getOp((itr.first)->getName());
        mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        set_attr(tensor_itr, itr.second);
      }
    }

    void dump(std::string const& file_name) const {
      FILE *fptr = fopen(file_name.c_str(), "w");
      if(fptr == nullptr) {
        throw mv::RuntimeError("RecomputeImplicitOpAttr",
          "Can't open file " + file_name);
      }

      std::string const message = "op=%s " + attr_name_ + "=%s\n";
      for (const auto& itr : attr_table_) {
        fprintf(fptr, message.c_str(),
              (itr.first)->getName().c_str(),
              attr_val_to_string(itr.second).c_str());
      }

      fclose(fptr);
    }

    size_t recompute() {
      op_list_t zid_nodes[2UL];
      zid_nodes[0UL] = zero_in_degree_nodes_;

      bool parity = false;
      while (!zid_nodes[parity].empty()) {
        op_list_t& curr_level = zid_nodes[parity];
        op_list_t& next_level = zid_nodes[!parity];

        for (const auto& zop : curr_level) {
          mv::Data::OpListIterator zop_itr = omodel_.getOp(zop->getName());
          for (auto citr = zop_itr.leftmostChild(); citr != omodel_.opEnd(); ++citr)
          {
            operation_t cop = &(*citr);
            auto ditr = in_degree_map_.find(cop);

            if ((ditr == in_degree_map_.end()) || (ditr->second == 0)) {
              throw mv::RuntimeError("LpScheduler", "in_degree_map_ invariant violation");
            }

            // maintain the inductive invariant //
            if (is_required_op_type(cop)) {
              auto const parent_attr = get_attr(zop);
              propagate_attr_to_child(cop, parent_attr);
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

      return attr_table_.size();
    }

  protected:
    std::string attr_name_; 
    mv::OpModel &omodel_;
    op_list_t& zero_in_degree_nodes_;
    degree_map_t in_degree_map_;
    table_t attr_table_;

    T get_attr_of_real_op(operation_t op_in) const {
      operation_non_const_t op = const_cast<operation_non_const_t>(op_in);
      if (!op->outputSlots()) { return T(); }
      mv::Data::TensorIterator tensor_itr = op->getOutputTensor(0UL);
      return get_attr_of_real_op(tensor_itr);
    }

    T get_attr(operation_t op) const {
      // inductive argument: any implicit op should have a valid attr
      if (is_required_op_type(op) &&
          (attr_table_.find(op) == attr_table_.end()) ) {
        throw mv::RuntimeError("LpScheduler", "Inductive invariant violation.");
      }

      return is_required_op_type(op) ?
        (attr_table_.find(op))->second :
        get_attr_of_real_op(op);
    }

    virtual void set_attr(mv::Data::TensorIterator tensor_itr, T const& val) const {
      tensor_itr->set<T>(attr_name_, val);
    };

    virtual std::string attr_val_to_string(T const&) const = 0;
    virtual bool is_required_op_type(operation_t const&) const = 0;
    virtual void propagate_attr_to_child(operation_t, T const&) = 0;
    virtual T get_attr_of_real_op(mv::Data::TensorIterator const &) const = 0;
}; // class Attrs<T> //

class Mem_Loc_Attr final: public Attrs<mem_location_t> {
  public:
    Mem_Loc_Attr(mv::OpModel& om, op_list_t& zero_in_degree, degree_map_t in_degree_map) :
      Attrs<mem_location_t>("Location", om, zero_in_degree, in_degree_map) {}

  private:
    void propagate_attr_to_child(operation_t child_op, mem_location_t const& parent_mem_loc) override {
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

    bool is_required_op_type(operation_t const& op) const override {
      return op->isImplicit();
    }

    virtual mem_location_t get_attr_of_real_op(mv::Data::TensorIterator const &tensor_itr) const override {
      return tensor_itr->get<mem_location_t>(attr_name_);
    }

}; // class Mem_Loc_Attr //

class Addr_Attr final: public Attrs<std::size_t> {
  typedef std::map<std::string, std::function<void(table_t&, operation_t, std::size_t const)>> func_map_t;

  public:
    Addr_Attr(mv::OpModel& om, op_list_t& zero_in_degree, degree_map_t in_degree_map) :
      Attrs<std::size_t>("address", om, zero_in_degree, in_degree_map) {}

  private:
    void propagate_attr_to_child(operation_t child_op, std::size_t const& parent_addr) override {
      // if already has an entry (for implicit ops with multiple inputs, ex. Concat)
      // make sure the address of the child is the lowest one

      auto propagate_parent_address = [] (table_t& attr_table, operation_t child, std::size_t const parent_address) {
        auto aitr = attr_table.find(child);
        if (aitr == attr_table.end()) {
          aitr = attr_table.insert(
              std::make_pair(child, parent_address)).first;
        } else {
          throw mv::RuntimeError("RecomputeImplicitOpAttr", "Implicit op " + child->getName() +
                " has more than one input");
        }
      };

      auto propagate_concat_address = [] (table_t& attr_table, operation_t child, std::size_t const parent_address) {
        auto aitr = attr_table.find(child);
        if (aitr == attr_table.end()) {
          aitr = attr_table.insert(
              std::make_pair(child, parent_address)).first;
        } else if (aitr->second > parent_address) {
          aitr->second = parent_address;
        }
      };

      auto propagate_slice_address = [] (table_t& attr_table, operation_t child, std::size_t const parent_address) {
        operation_non_const_t op = const_cast<operation_non_const_t>(child);
        auto const& start = child->get<mv::Shape>("begin");
        auto const& in_tensor = op->getInputTensor(0);
        auto const& in_shape = in_tensor->getShape();
        auto const indices = in_tensor->getOrder().getContiguityVector();

        std::size_t multiplier = in_tensor->getDType().getSizeInBytes();
        std::size_t offset = start[indices[0]] * multiplier;
        for (std::size_t idx = 1; idx < in_shape.ndims(); ++idx) {
          multiplier *= in_shape[indices[idx - 1]];
          offset += start[indices[idx]] * multiplier;
        }

        auto aitr = attr_table.find(child);
        if (aitr == attr_table.end()) {
          aitr = attr_table.insert(
              std::make_pair(child, parent_address + offset)).first;
        } else {
          throw mv::RuntimeError("RecomputeImplicitAttr", "Implicit op " + child->getName() +
                " has more than one input");
        }
      };

      func_map_t propagate_addr = {
        {"Align", propagate_parent_address},
        {"Copy", propagate_parent_address},
        {"Crop", propagate_parent_address},
        {"ImplicitConcat", propagate_concat_address},
        {"ImplicitPermute", propagate_parent_address},
        {"ImplicitReshape", propagate_parent_address},
        {"Slice", propagate_slice_address}
      };

      propagate_addr.at(child_op->getOpType())(attr_table_, child_op, parent_addr);

    }

    std::string attr_val_to_string(std::size_t const& val) const override {
      return std::to_string(val);
    }

    // TODO: Add addres computation for the other implicit ops (ImplicitOutput,
    // ImplicitUnion, ImplicitInput, ImplicitInputSlice, ImplicitJoin)
    virtual bool is_required_op_type(operation_t const& op) const override {
      const std::vector<std::string> reqOps = { 
        "Align", "Copy", "Crop", "ImplicitConcat",
        "ImplicitPermute", "ImplicitReshape", "Slice" };
      return std::find(reqOps.begin(), reqOps.end(), op->getOpType()) != reqOps.end();
    }

    virtual void set_attr(mv::Data::TensorIterator tensor_itr, std::size_t const& val) const override {
      tensor_itr->setAddress(val);
    };

    virtual std::size_t get_attr_of_real_op(mv::Data::TensorIterator const &tensor_itr) const override {
      if (tensor_itr->hasAttr(attr_name_)) {
        return tensor_itr->get<std::size_t>(attr_name_);
      }

      mv::DataModel dm(omodel_);
      auto tensorAllocators = tensor_itr->get<std::set<std::string>>("allocators");
      if (tensorAllocators.empty())
          throw mv::ArgumentError("buildTensorReferenceT", "",  "Tensor Allocators empty", "");

      auto tensorAllocator = dm.getAllocator(*tensorAllocators.begin());
      mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, tensor_itr); // 0 is the only stage for now, but this will probably change in the future
      auto masterBuffer = tensorAllocator.getTopMasterBuffer(tensorBufferIt);
      return (*masterBuffer)->getOffset();
    }
}; // class Addr_Attr //

static void RecomputeImplicitOpAttr(const mv::pass::PassEntry&,
    mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv {
namespace pass {

MV_REGISTER_PASS(RecomputeImplicitOpAttr)
  .setFunc(RecomputeImplicitOpAttr);

} // namespace pass //
} // namespace mv//

void RecomputeImplicitOpAttr(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc,
      mv::Element&) {
  mv::OpModel om(model);
  op_list_t zero_in_degree_nodes;
  degree_map_t initial_in_degree_map;

  compute_ops_in_degree(om, zero_in_degree_nodes, initial_in_degree_map);

  auto const attr = passDesc.get<std::string>("attribute");

  if (attr == "Location") {
    Mem_Loc_Attr memLocationAttribute(om, zero_in_degree_nodes, initial_in_degree_map);
    memLocationAttribute.recompute();
    memLocationAttribute.set_recomputed_attr();

    if (passDesc.hasAttr("output_dir")) {
      std::string output_dir = passDesc.get<std::string>("output_dir");
      memLocationAttribute.dump(output_dir + "/recompute_mem_loc.txt");
    }
  }

  if (attr == "address") {
    Addr_Attr addressAttribute(om, zero_in_degree_nodes, initial_in_degree_map);
    addressAttribute.recompute();
    addressAttribute.set_recomputed_attr();

    if (passDesc.hasAttr("output_dir")) {
      std::string output_dir = passDesc.get<std::string>("output_dir");
      addressAttribute.dump(output_dir + "/recompute_address.txt");
    }
  }
}
