#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <math.h>

static void generateEltWiseConstantsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateEltWiseConstants)
        .setFunc(generateEltWiseConstantsFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );
    }
}


// QUESTION: Is this pass really needed or it's just wasting space in our precious CMX????
static void generateEltWiseConstantsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto eltWiseDpuTaskOp = om.opBegin(); eltWiseDpuTaskOp != om.opEnd(); ++eltWiseDpuTaskOp)
    {
        if(eltWiseDpuTaskOp->getOpType() == "DPUTask")
        {
            if(eltWiseDpuTaskOp->get<std::string>("taskOp") == "Eltwise")
            {
                bool hasBias = eltWiseDpuTaskOp->hasAttr("bias");
                if (hasBias)
                {
                    std::string opName = eltWiseDpuTaskOp->getName();

                    std::string name(opName + "_bias");
                    auto output = eltWiseDpuTaskOp->getOutputTensor(0);
                    auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

                    mv::Shape shape({1, 1, 1, outputChannels});
                    std::vector<int64_t> constantData(shape.totalSize(), 0);

                    auto bias = dm.getTensor(eltWiseDpuTaskOp->get<std::string>("bias"));
                    auto biasData = bias->getData(); //Bias has the type Int32 in both cases above

                    for (size_t i = 0; i < constantData.size(); ++i)
                        constantData[i] = biasData[i/4];

                    dm.undefineTensor(bias);
                    eltWiseDpuTaskOp->erase("bias");

                    mv::QuantizationParams quantParams = {{},{},{},{}};
                    auto constant = om.constantInt(constantData, shape, mv::DType("Int32"), mv::Order("NWCH"), quantParams, name);
                    om.getSourceOp(constant)->set<unsigned>("opId", eltWiseDpuTaskOp->get<unsigned>("opId"));
                    unsigned newSize = eltWiseDpuTaskOp->addInputTensor(constant);
                    om.defineFlow(constant, eltWiseDpuTaskOp, newSize - 1);
                }
            }
        }
    }
}
