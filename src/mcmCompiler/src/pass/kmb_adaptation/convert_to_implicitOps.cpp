#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void convertToImplicitOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);


namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvertToImplicitOps)
            .setFunc(convertToImplicitOpsFcn)
            .setDescription(
                "Replaces Permute,Reshape, Resample ops with implicit ops");
    }
}


void convertToImplicitOpsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    std::vector<std::string> opList = {"Permute", "Reshape", "Resample"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operations = om.getOpsOfTypes(opList);
    std::vector <mv::Data::OpListIterator> ops;
    ops.reserve(operations["Permute"].size() + operations["Reshape"].size() + operations["Resample"].size() );
    ops.insert(ops.end(), operations["Permute"].begin(), operations["Permute"].end());
    ops.insert(ops.end(), operations["Reshape"].begin(), operations["Reshape"].end());
    ops.insert(ops.end(), operations["Resample"].begin(), operations["Resample"].end());

    for (auto& opIt : ops)
    {
        auto input = opIt->getInputTensor(mv::IO_TENSOR_INPUT);
        auto output = opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        auto input_shape = input->getShape();
        auto output_shape = output->getShape();
        auto opType = opIt->getOpType();
        mv::Tensor::MemoryLocation outputLocation;
        if(output->hasAttr("Location"))
        {
            outputLocation = output->get<mv::Tensor::MemoryLocation>("Location");
        }

        if(opType == "Reshape")
        {
            output_shape = opIt->get<mv::Shape>("shape");
        }
        auto is_explicit = true;

        if(opType == "Permute" || opType == "Reshape")
        {
            // If input & output are both 1D, permute/Reshape can be implicit
            if (input_shape.isFlat() && output_shape.isFlat())
                is_explicit = false;

           // Check if forced to implicit
           if (opIt->hasAttr("isImplicit") && opIt->get<bool>("isImplicit"))
               is_explicit = false;

            // Skip if explicit
            if (is_explicit)
                continue;            

            // Avoid unnecessary DMAs
            if(input->hasAttr("Location"))
                outputLocation = input->get<mv::Tensor::MemoryLocation>("Location");
        }

        if(opType == "Resample")
        {
            // Skip if explicit
            if (!(opIt->hasAttr("isImplicit") && opIt->get<bool>("isImplicit")))
                continue;
            //resample op marked as implicit is always followed by identity Convolution
            //output location of resample op is DDR as it is UPA task, which can add unecessary DMA operation
            //hence for ImplicitResample outputLocation is set as per input location of resample op, which can avoid DMA ops
            if(input->hasAttr("Location"))
            {
                outputLocation = input->get<mv::Tensor::MemoryLocation>("Location");
            }
        }

        // Replace ops with implicit ops
        std::string splitStrategy;
        if(opIt->hasAttr("splitStrategy"))
            splitStrategy = opIt->get<std::string>("splitStrategy");
        auto quantParams = output->getQuantParams();
        auto opId = opIt->get<unsigned>("opId");
        auto explicitStrides = opIt->hasAttr("explicitStrides") && opIt->get<bool>("explicitStrides");
        auto forceU8 = opIt->hasAttr("forceU8") && opIt->get<bool>("forceU8");

        auto outputFlows = mv::getOutputDataFlow(om, opIt);

        mv::Data::TensorIterator implicitOp;
        if(opType == "Permute")
        {
            auto dtype = input->getDType();
            implicitOp = om.implicitPermute("", input, output_shape);
            implicitOp->setDType(dtype);
        }
        else if(opType == "Reshape")
        {
            implicitOp = om.implicitReshape("", input, output_shape);
        }
        else if(opType == "Resample")
        {
            implicitOp = om.implicitResample("", input, output_shape);
            // Store input shapes, used later to compute SEP table offsets
            om.getSourceOp(implicitOp)->set<mv::Shape>("originalShape", input_shape);            
        }
        implicitOp->setQuantParams(quantParams);
        om.getSourceOp(implicitOp)->set<unsigned>("opId", opId);
        implicitOp->set<mv::Tensor::MemoryLocation>("Location", outputLocation);

        if (explicitStrides)
            om.getSourceOp(implicitOp)->set<bool>("explicitStrides", true);

        if (forceU8)
        {
            om.getSourceOp(implicitOp)->set<bool>("forceU8", true);
            implicitOp->setDType(mv::DType("UInt8"));
        }

        if(!splitStrategy.empty())
            om.getSourceOp(implicitOp)->set<std::string>("splitStrategy", splitStrategy);

        mv::setOutputDataFlow(om, implicitOp, outputFlows);

        auto parentInputTensor = om.getSourceOp(implicitOp)->getInputTensor(mv::IO_TENSOR_INPUT);
        if(outputLocation == mv::Tensor::MemoryLocation::OUTPUT)
        {
            //last op
            parentInputTensor->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        }
    }
}
