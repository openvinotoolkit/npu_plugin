#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/base_op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

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

double tileEfficiency(mv::Data::OpListIterator opIt, bool SOH, bool mcm = true)
{
    auto output = opIt->getOutputTensor(0)->getShape();
    size_t K = output[mv::IO_CHANNEL_DIMENSION];
    size_t H = output[mv::IO_HEIGHT_DIMENSION];
    size_t W = output[mv::IO_WIDTH_DIMENSION];
    auto KHW = K * H * W;

    auto denom1 = mv::round_up(H, 4) * mv::round_up(W, 4) * mv::round_up(K, 320);
    double opt1 = (double) KHW / (double) denom1;

    auto denom2 = mv::round_up(H, 80) * mv::round_up(W, 1) * mv::round_up(K, 64);
    double opt2 = (double) KHW / (double) denom2;

    auto denom3 = mv::round_up(H, 20) * mv::round_up(W, 4) * mv::round_up(K, 64);
    double opt3 = (double) KHW / (double) denom3;

    auto denom4 = mv::round_up(H, 16) * mv::round_up(W, 1) * mv::round_up(K, 320);
    double opt4 = (double) KHW / (double) denom4;

    auto denom5 = mv::round_up(H, 16) * mv::round_up(W, 20) * mv::round_up(K, 16);
    double opt5 = (double) KHW / (double) denom5;

    auto denom6 = mv::round_up(H, 64) * mv::round_up(W, 5) * mv::round_up(K, 16);
    double opt6 = (double) KHW / (double) denom6;

    double max1 = std::max(opt1, opt2);
    double max2 = std::max(opt3, opt4);
    double max3 = std::max(max1, max2);
    double max4 = std::max(opt5, opt6);

    if(mcm && SOH)
    {
        return max4;
    }
    if(mcm && !SOH)
    {
        return max3;
    }
    if(!mcm) //pick whichever is best, regardless of strategy chosen
    {
        return std::max(max3, max4);
    }

    return 1.0;
}

bool calcComputeOrDataBound(mv::Data::OpListIterator opIt, double macs, double bytes, bool mcm)
{
    if(!opIt->hasAttr("splitStrategy")) return false;

    auto clusteringStrategy = opIt->get<std::string>("splitStrategy");
    double tiles = 20;
    if(clusteringStrategy == "Clustering")
        tiles = 5;

    double fp16Mult = 1.0;
    if(opIt->getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16"))
        fp16Mult = 0.25;

    double tileEff = 1.0;
    if(clusteringStrategy == "SplitOverH")
        tileEff = tileEfficiency(opIt, true);
    else
        tileEff = tileEfficiency(opIt, false);

    if(!mcm)
        tileEff = tileEfficiency(opIt, true, false);


    //MACs / (256 * fp16Mult * tiles * tileEff)   / 700
    auto computeTime = (macs / ((256*fp16Mult * tiles) * tileEff)) / 700;
    // bytes / BW / 700
    auto dmaTime = bytes / 28.5714 / 700;

    // std::cout << opIt->getName() << ": " << computeTime << " ? " << dmaTime << std::endl;
    // std::cout << "     " << macs << " / (" << (256*fp16Mult*tiles) << " * " << tileEff << ")  /  " << 700 << std::endl; 

    if(computeTime > dmaTime)
        return true;

    return false;
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
    long long int totalMacs = 0;
    long long int computeBoundMacs = 0;
    size_t computeBoundLayers = 0;
    size_t dataBoundLayers = 0;
    long long int dataBoundBytes = 0;
    size_t networkComputeBoundLayers = 0;
    size_t networkDataBoundLayers = 0;

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

                size_t baseKernelCost = 1;
                long long int bytes = 0;
                if (opType == "MaxPool")
                {
                    auto kernel = opIt->get<std::array<unsigned short,2>>("kSize");
                    baseKernelCost = kernel[0] * kernel[1];
                }
                else if ((opType == "DepthwiseConv") || (opType == "Conv"))
                {
                    auto weights = opIt->getInputTensor(1);
                    auto weightsShape = weights->getShape();
                    baseKernelCost = weightsShape[mv::KERNEL_WIDTH] * weightsShape[mv::KERNEL_HEIGHT];

                    if(opType == "Conv")
                        baseKernelCost *= weightsShape[mv::KERNEL_INPUT_CHANNELS];

                    bytes += weights->computeTotalSize();
                }

                long long int macs = (baseKernelCost * opIt->getOutputTensor(0)->computeTotalSize());
                totalMacs += macs;

                auto totalActSize = getActivationSizes(opIt);
                if(totalActSize < 917504)
                    DPUactsLessCMX++;
                else if(totalActSize >= (917504*4))
                {
                    DPUactsMoreCMX++;
                    bytes += totalActSize;
                }
                else
                {
                    if(opIt->hasAttr("splitStrategy") && opIt->get<std::string>("splitStrategy") == "Clustering")
                        bytes += totalActSize;
                    DPUactsBtwnCMX++;
                }


                bool isComputeBound = calcComputeOrDataBound(opIt, macs, bytes, true);
                if(isComputeBound)
                {
                    computeBoundMacs += macs;
                    computeBoundLayers++;
                }
                else
                {
                    dataBoundBytes += bytes;
                    dataBoundLayers++;
                }

                bool isNetComputeBound  = calcComputeOrDataBound(opIt, macs, bytes, false);
                if(isNetComputeBound)
                {
                    networkComputeBoundLayers++;
                }
                else{
                    networkDataBoundLayers++;
                }
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

    if(numOutputs > 1)  numOutputs--; // If we have mutliple outputs, just count the implicits, not the final output

    // if(base)
    // {
    //     std::cout << "Is Classification: " << std::boolalpha << isClassificationNet << std::endl;
    //     std::cout << "Total Parallel branches: " << totalBranches << std::endl;
    //     std::cout << "Max Parallel branches: " << maxBranches << std::endl;
    //     std::cout << "Input size: " << sizeInput << std::endl;
    //     std::cout << "Input shape: " << inputShape.toString() << std::endl;
    //     std::cout << "Outputs: " << numOutputs << std::endl;
    //     std::cout << "U8 DPU Tasks: " << numDPUtasks << std::endl;
    //     std::cout << "FP16 DPU Tasks: " << numFP16tasks << std::endl;
    //     std::cout << "Total MACs: " << totalMacs << std::endl;
    //     std::cout << "DPU < CMX: " << DPUactsLessCMX << std::endl;
    //     std::cout << "DPU > CMX: " << DPUactsMoreCMX << std::endl;
    //     std::cout << "DPU ~ CMX: " << DPUactsBtwnCMX << std::endl;
    //     std::cout << "MCM Compute-Bound MACs:" << computeBoundMacs << std::endl;
    //     std::cout << "MCM Data-Bound Bytes:" << dataBoundBytes << std::endl;
    //     std::cout << "MCM Compute-Bound Layers:" << computeBoundLayers << std::endl;
    //     std::cout << "MCM Data-Bound Layers:" << dataBoundLayers << std::endl;
    //     std::cout << "Topo Compute-Bound Layers:" << networkComputeBoundLayers << std::endl;
    //     std::cout << "Topo Data-Bound Layers:" << networkDataBoundLayers << std::endl;
    // }
    // else
    // {
    //     std::cout << "Total UPA Tasks: " << numSWlayers << std::endl;
    //     std::cout << "Intermediate UPA Tasks: " << numIntermSWlayers << std::endl;
    //     std::cout << "UPA < CMX: " << UPAactsLessCMX << std::endl;
    //     std::cout << "UPA > CMX: " << UPAactsMoreCMX << std::endl;
    //     std::cout << "UPA ~ CMX: " << UPAactsBtwnCMX << std::endl;
    //     std::cout << "DPU tasks tiled: " << numTiledDPUtasks << std::endl;
    //     std::cout << "DMAs to CMX: " << dmaToCMX << std::endl;
    //     std::cout << "Bytes moved to CMX: " << sizeToCMX << std::endl; 
    //     std::cout << "DMAs to DDR: " << dmaToDDR << std::endl;
    //     std::cout << "Bytes moved to DDR: " << sizeToDDR << std::endl; 

    // }

    if(base)
    {
        std::cout << isClassificationNet << std::endl;
        std::cout << totalBranches << std::endl;
        std::cout << maxBranches << std::endl;
        std::cout << sizeInput << std::endl;
        std::cout << inputShape.toString() << std::endl;
        std::cout << numOutputs << std::endl;
        std::cout << numDPUtasks << std::endl;
        std::cout << numFP16tasks << std::endl;
        std::cout << totalMacs << std::endl;
        std::cout << DPUactsLessCMX << std::endl;
        std::cout << DPUactsMoreCMX << std::endl;
        std::cout << DPUactsBtwnCMX << std::endl;
            std::cout << computeBoundMacs << std::endl;
            std::cout << dataBoundBytes << std::endl;
            std::cout <<  computeBoundLayers << std::endl;
            std::cout <<  dataBoundLayers << std::endl;
            std::cout << networkComputeBoundLayers << std::endl;
            std::cout <<  networkDataBoundLayers << std::endl;
    }
    else
    {
        std::cout << numSWlayers << std::endl;
        std::cout << numIntermSWlayers << std::endl;
        std::cout << UPAactsLessCMX << std::endl;
        std::cout << UPAactsMoreCMX << std::endl;
        std::cout << UPAactsBtwnCMX << std::endl;
        std::cout << numTiledDPUtasks << std::endl;
        std::cout << dmaToCMX << std::endl;
        std::cout << sizeToCMX << std::endl; 
        std::cout << dmaToDDR << std::endl;
        std::cout << sizeToDDR << std::endl; 

    }
}