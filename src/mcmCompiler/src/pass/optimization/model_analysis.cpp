#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/base_op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void modelAnalysisFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ModelAnalysis)
            .setFunc(modelAnalysisFcn)
            .setDescription(
                "Gather stats about this network");

    }
}

size_t getActivationSizes(mv::Data::OpListIterator opIt)
{
    size_t activations = opIt->getOutputTensor(0)->computeTotalSize();
    activations += opIt->getInputTensor(0)->computeTotalSize();
    if(opIt->getOpType() == "Eltwise")
    {
        activations += opIt->getInputTensor(1)->computeTotalSize();
    }
    return activations;
}

bool onlyOneDim(mv::Data::OpListIterator opIt)
{
    bool alreadyFound = false;
    auto tensorShape = opIt->getInputTensor(0)->getShape();
    for(size_t dim = 0; dim < tensorShape.ndims(); dim++)
        if(tensorShape[dim] > 1)
        {
            if(alreadyFound) return false;
            alreadyFound = true;
        }

    return true;
}

int countInputLayers( mv::ComputationModel& model, mv::Data::OpListIterator op){
    mv::OpModel om(model);
    int inputs = 0;
    if(op->getOpType() == "Input") inputs = 1;
    for(auto inputOp = op.leftmostParent(); inputOp != om.opEnd(); ++inputOp)
    {
        auto inputType = inputOp->getOpType();
        if ((inputType == "Constant") ||
        (inputType == "ConstantInt") ||
        (inputType == "ConstantDataElement") ||
        (inputType == "WeightsTable") ||
        (inputType == "SparsityMap"))
            continue;
        inputs++;
    }
    return inputs;
}


void modelAnalysisFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    bool dma = passDesc.hasAttr("dma") ? passDesc.get<bool>("dma"): false;
    bool base = !dma;

    mv::OpModel om(model);
    mv::DataModel dm(model);
    size_t numOutputs = 0;
    size_t sizeInput = 0;
    mv::Shape inputShape;
    size_t numDPUtasks = 0;
    size_t numTiledDPUtasks = 0;
    size_t numSWlayers = 0;
    size_t numIntermSWlayers = 0;
    size_t numFP16tasks = 0;
    size_t DPUactsLessCMX = 0;
    size_t DPUactsMoreCMX = 0;
    size_t DPUactsBtwnCMX = 0;
    size_t UPAactsLessCMX = 0;
    size_t UPAactsMoreCMX = 0;
    size_t UPAactsBtwnCMX = 0;
    bool isClassificationNet = false;
    size_t dmaToCMX = 0;
    long long int sizeToCMX = 0;
    size_t dmaToDDR = 0;
    long long int sizeToDDR = 0;
    size_t totalBranches = 1;
    size_t maxBranches = 1;

    if(base){
        std::vector<mv::Data::OpListIterator> ops = om.topologicalSort();
        size_t runningBranches = 1;
        for(auto opIt : ops)
        {
            auto opType = opIt->getOpType();
            // Skip weights/constants nodes (have already been added to convs/software layers at this point)
            if ((opType == "Constant") ||
                (opType == "ConstantInt") ||
                (opType == "ConstantDataElement") ||
                (opType == "WeightsTable") ||
                (opType == "SparsityMap") ||
                (opType == "Input") ||
                (opType == "Output") )
                continue;

            // Get number of input and output to node
            int inputs = countInputLayers(om, opIt);
            int outputs = opIt.childrenSize();
            if(outputs > 1)
                totalBranches += outputs - 1;

            runningBranches += outputs - 1;
            runningBranches -= (inputs-1);

            if(runningBranches > maxBranches)
                maxBranches = runningBranches;
        }
    }

    std::vector<mv::Data::OpListIterator> ops = om.getOps();
    for(auto opIt : ops)
    {
        auto opType = opIt->getOpType();
        if(base){
            if(opType == "Input")
            {
                sizeInput = opIt->getOutputTensor(0)->computeTotalSize();
                inputShape = opIt->getOutputTensor(0)->getShape();
            }
            if(opType == "Output" || opType == "ImplicitOutput")
            {    
                numOutputs++;
                if(opType == "Output" && onlyOneDim(opIt))
                    isClassificationNet = true;
            }
            if(opIt->isHardwarizable() && !(opIt->hasAttr("softwareExecuted") && opIt->get<bool>("softwareExecuted")))
            {
                if(opIt->getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16"))
                    numFP16tasks++;
                else
                    numDPUtasks++;
                
                auto totalActSize = getActivationSizes(opIt);
                if(totalActSize < 917504)
                    DPUactsLessCMX++;
                else if(totalActSize >= (917504*4))
                    DPUactsMoreCMX++;
                else
                    DPUactsBtwnCMX++;
            }

        }
        if(dma){
            if(opType == "DMATask")
            {
                auto direction = opIt->get<mv::DmaDirection>("direction");
                if(direction == mv::NNCMX2DDR)
                {
                    dmaToDDR++;
                    sizeToDDR += opIt->getOutputTensor(0)->computeTotalSize();
                }
                if(direction == mv::DDR2NNCMX)
                {    
                    dmaToCMX++;
                    sizeToCMX += opIt->getOutputTensor(0)->computeTotalSize();
                }
            }
            if(opType == "DPUTask")
                numTiledDPUtasks++;
            
            if(opType == "UPATask" || opIt->isUPA() || (opIt->hasAttr("softwareExecuted") && opIt->get<bool>("softwareExecuted")))
            {
                numSWlayers++;
                auto totalActSize = getActivationSizes(opIt);
                if(totalActSize < 917504)
                    UPAactsLessCMX++;
                else if(totalActSize >= (917504*4))
                    UPAactsMoreCMX++;
                else
                    UPAactsBtwnCMX++;

                // Check if this guy is intermediate, which we will define as having a following DPU task between it and output
                auto nextOp = mv::findSinkLayers(dm, opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT))[0];
                while(!(nextOp->getOpType() == "ImplicitOutput" || nextOp->getOpType() == "Output"))
                {
                    if(nextOp->isHardwarizable())
                    {
                        numIntermSWlayers++;
                        break;
                    }
                    nextOp = mv::findSinkLayers(dm, nextOp->getOutputTensor(mv::IO_TENSOR_OUTPUT))[0];
                }
            }
        }
    }

    if(base)
    {
        std::cout << "Is Classification: " << std::boolalpha << isClassificationNet << std::endl;
        std::cout << "Total Parallel branches: " << totalBranches << std::endl;
        std::cout << "Max Parallel branches: " << maxBranches << std::endl;
        std::cout << "Input size: " << sizeInput << std::endl;
        std::cout << "Input shape: " << inputShape.toString() << std::endl;
        std::cout << "Outputs: " << numOutputs << std::endl;
        std::cout << "U8 DPU Tasks: " << numDPUtasks << std::endl;
        std::cout << "FP16 DPU Tasks: " << numFP16tasks << std::endl;
        std::cout << "DPU < CMX: " << DPUactsLessCMX << std::endl;
        std::cout << "DPU > CMX: " << DPUactsMoreCMX << std::endl;
        std::cout << "DPU ~ CMX: " << DPUactsBtwnCMX << std::endl;
    }
    else
    {
        std::cout << "Total UPA Tasks: " << numSWlayers << std::endl;
        std::cout << "Intermediate UPA Tasks: " << numIntermSWlayers << std::endl;
        std::cout << "UPA < CMX: " << UPAactsLessCMX << std::endl;
        std::cout << "UPA > CMX: " << UPAactsMoreCMX << std::endl;
        std::cout << "UPA ~ CMX: " << UPAactsBtwnCMX << std::endl;
        std::cout << "DPU tasks tiled: " << numTiledDPUtasks << std::endl;
        std::cout << "DMAs to CMX: " << dmaToCMX << std::endl;
        std::cout << "Bytes moved to CMX: " << sizeToCMX << std::endl; 
        std::cout << "DMAs to DDR: " << dmaToDDR << std::endl;
        std::cout << "Bytes moved to DDR: " << sizeToDDR << std::endl; 

    }
}