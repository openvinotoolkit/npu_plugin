#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/pass_quantization.hpp"
#include <math.h>

static void resolveFQAfterBranchingSlicesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(ResolveFQAfterBranchingSlices)
        .setFunc(resolveFQAfterBranchingSlicesFcn)
        .setDescription(
            "This pass solves cases where a quantizable op's output goes into multiple slice layers that have FQ layers as children"
        );
    }
}

static double inf = std::numeric_limits<double>::infinity();

// This is needed to avoid a static initializer fiasco.
static const mv::QuantizationParams& initial_quant_params()
{
    static mv::QuantizationParams init{{0}, {1}, {-inf}, {inf}};
    return init;
}

static bool isQuantizableOp(mv::Data::OpListIterator op)
{
    static std::set<std::string> quantizable_ops{"Conv", "FullyConnected", "Eltwise" , "AveragePool", "DepthwiseConv", "Scale"};
    return quantizable_ops.count(op->getOpType());
}

void placeReQuantizeDepthwiseBefore(
    mv::OpModel & om, mv::Data::OpListIterator opIt,
    std::size_t index, mv::Data::TensorIterator inputTensor,
    mv::QuantizationParams quantParams)
{
    //FIND THE APPROPRIATE FLOW
    auto inputFlow = opIt.leftmostInput();
    while(inputFlow != om.flowEnd())
    {
        auto tensor = inputFlow->getTensor();
        if (tensor->getName() == inputTensor->getName())
        {
            break;
        }
        ++inputFlow;
    }
    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = { 0 };
    std::vector<double> min = { 1 };
    std::vector<double> max = { 1 };

    std::vector<double> scale(1, 1.0f);
    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    int64_t weightsValue = 1;
    std::vector<int64_t> weightsData(inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
    weights = om.constantInt(opIt->getName() + "Depthwise_requant_wt",
                        weightsData,
                        {1, 1, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
                        getDType(mv::Precision::U8),
                        mv::Order(mv::Order::getRowMajorID(4)));
    weights->setQuantParams(weightsQuantParams);
    auto reQuantizeDepthwise = om.depthwiseConv(
        opIt->getName() + "Depthwise_requant" + std::to_string(index),
        inputTensor, weights, {1,1}, {0, 0, 0, 0}, 1);
    reQuantizeDepthwise->setQuantParams(quantParams);
    reQuantizeDepthwise->setDType(mv::DType("UInt8"));
    auto reQuantizeDepthwiseOp = om.getSourceOp(reQuantizeDepthwise);
    auto weightsOp = om.getSourceOp(weights);
    reQuantizeDepthwiseOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
    weightsOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
    om.undefineFlow(inputFlow);
    opIt->setInputTensor(reQuantizeDepthwise, index, false);
    om.defineFlow(reQuantizeDepthwise, opIt, index);
}

void addFQAfter(mv::OpModel& om, const mv::Data::TensorIterator& sourceTensor, const mv::Data::OpListIterator& fqOp)
{
    if (fqOp->getOpType() != "FakeQuantize")
    {
        throw mv::RuntimeError("ResolveFQAfterBranchingSlices",
            "op " + fqOp->getName() + " should be FQ, but isn't. (" + fqOp->getOpType() + ")");
    }

    mv::DataModel dm(om);
    auto children = findSinkLayers(dm, sourceTensor);

    auto makeConstant = [&om] (mv::Data::TensorIterator constData, unsigned opId) -> mv::Data::TensorIterator
    {
        auto constTensor = om.constant(
            om.getSourceOp(constData)->getName()+ ":fq_data:" + std::to_string(opId),
            constData->getDoubleData(),
            constData->getShape(),
            constData->getDType(),
            constData->getOrder());
        auto constOp = om.getSourceOp(constTensor);
        constOp->set<unsigned>("opId", opId);
        return constTensor;
    };

    auto inputMin = makeConstant(fqOp->getInputTensor(1), fqOp->get<unsigned>("opId"));
    auto inputMax = makeConstant(fqOp->getInputTensor(2), fqOp->get<unsigned>("opId"));
    auto outputMin = makeConstant(fqOp->getInputTensor(3), fqOp->get<unsigned>("opId"));
    auto outputMax = makeConstant(fqOp->getInputTensor(4), fqOp->get<unsigned>("opId"));

    auto fq = om.fakeQuantize(fqOp->getName() + "_star", sourceTensor, inputMin, inputMax, outputMin,
                              outputMax, fqOp->get<unsigned>("levels"));
    auto newFqOp = om.getSourceOp(fq);
    newFqOp->set<unsigned>("opId", fqOp->get<unsigned>("opId"));

    for (auto& child : children)
    {
        std::size_t inIndex = 0;
        auto inputFlow = child.leftmostInput();
        for(; inputFlow != om.flowEnd(); ++inputFlow, ++inIndex)
        {
            const auto& tensor = inputFlow->getTensor();
            if (tensor->getName() == sourceTensor->getName())
            {
                break;
            }
        }

        om.undefineFlow(inputFlow);
        child->setInputTensor(fq, inIndex, false);
        om.defineFlow(fq, child, inIndex);
    }
}

/*
    Iterate over all the parents and try to move FakeQuantize before the Slice layers
    If the quant params are not the same on all branches, move the most common FQ before
    Slice and requantize the other branches with DW Conv.
    Ex: Let FQ_0 be the selected FQ op,
         | --- Slice_0 --- FQ_0                  | --- Slice_0 --- ...
    Op --|                      => Op --- FQ_0 --|
         | --- Slice_1 --- FQ_1                  | --- Slice_1 --- DW Conv --- FQ_1 --- ...
*/
void moveFQBeforeSlice(mv::DataModel& dm, const mv::Data::TensorIterator& parent)
{
    mv::OpModel om(dm);
    auto children = findSinkLayers(dm, parent);

    std::vector<mv::QuantizationParams> quantParams;
    std::vector<mv::Data::OpListIterator> grandchildren(children.size());
    std::vector<int> bins(children.size(), -1);
    std::vector<int> numOccurences;
    int crtBin = 0;
    for (std::size_t childIdx = 0; childIdx < children.size(); childIdx++)
    {
        auto nextOps = findSinkLayers(dm, children.at(childIdx)->getOutputTensor(0));

        // If one of the Slice Ops branches out, do nothing.
        if (nextOps.size() != 1)
            return;

        // For FQ ops, extract output quant params.
        // If first occurence, add to quantParams vec, else increase occurence
        // count for that set of quant params.
        if (nextOps.at(0)->getOpType() == "FakeQuantize")
        {
            const auto& crtQuantParams = extractQuantParamsO(nextOps.at(0), true);

            const auto res = std::find_if(quantParams.begin(), quantParams.end(),
                                          [&](mv::QuantizationParams& quant) {
                                                return isEqual(quant, crtQuantParams);
                                        });
            if (res == quantParams.end())
            {
                quantParams.emplace_back(crtQuantParams);
                bins.at(childIdx) = crtBin++;
                numOccurences.emplace_back(1);
            } else {
                const int binIdx = std::distance(quantParams.begin(), res);
                bins.at(childIdx) = binIdx;
                numOccurences.at(binIdx)++;
            }
        }

        grandchildren.at(childIdx) = nextOps.at(0);
    }

    // There's no Slice Op with FQ child
    if (numOccurences.empty())
        return;

    const std::size_t idxMostCommonQuantParam = std::distance(
        numOccurences.begin(),
        std::max_element(numOccurences.begin(), numOccurences.end()));

    // Add FQ between parent op and the slice children
    const std::size_t idxSelFQ = std::distance(bins.begin(), std::find(bins.begin(), bins.end(), idxMostCommonQuantParam));
    addFQAfter(om, parent, grandchildren.at(idxSelFQ));

    for (std::size_t idx = 0; idx < grandchildren.size(); ++idx)
    {
        auto grandchild = grandchildren.at(idx);
        auto grandchildInputs = grandchild->getInputTensor();

        // If the grandchild is the FQ op that was moved before the slices,
        // remove it from the model.
        if (bins.at(idx) == static_cast<int>(idxMostCommonQuantParam))
        {
            if (grandchild->getOpType() != "FakeQuantize")
            {
                throw mv::RuntimeError("ResolveFQAfterBranchingSlices",
                    "op " + grandchild->getName() + " should be FQ, but isn't. (" + grandchild->getOpType() + ")");
            }

            linkNewOperationsRemove(children.at(idx),
                grandchildInputs.at(0),
                om, grandchild);
            continue;
        }

        // If the grandchild is a quantizable op, don't add requantization DW.
        if (isQuantizableOp(grandchild))
            continue;

        // In all other cases, add a requantization DW conv between the Slice Op
        // and its descendant.
        std::size_t inputIdx = 0;
        for (; inputIdx < grandchildInputs.size(); ++inputIdx)
        {
            if (children.at(idx)->getOutputTensor(0)->getName() == grandchildInputs.at(inputIdx)->getName())
                break;
        }

        placeReQuantizeDepthwiseBefore(
            om, grandchild, inputIdx, children.at(idx)->getOutputTensor(0), initial_quant_params());
    }
}

/* 
    Treats cases where Slice Ops with same parent are followed by FQ
    Motivation: In cases such as the one below, the op with quantizable output will remain
    unquantized, because the first op after branching is not a FQ Op
    Ex:
                       | --- Slice --- FQ
    (Quantizable Op) --|
                       | --- Slice --- FQ
*/
void resolveFQAfterBranchingSlicesFcn(const mv::pass::PassEntry& /*pass*/, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS);

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto alreadyInVec = [](const std::vector<mv::Data::TensorIterator>& tensorVec, const mv::Data::TensorIterator& tensor) -> bool {
        return std::find(tensorVec.cbegin(), tensorVec.cend(), tensor) != tensorVec.cend();
    };

    auto sliceOps = om.getOps("Slice");
    std::vector<mv::Data::TensorIterator> sliceInputs;
    sliceInputs.reserve(sliceOps.size());
    for (auto slice = sliceOps.begin(); slice != sliceOps.end(); slice++)
    {
        auto inputTensor = (*slice)->getInputTensor()[0]; 
        auto prevOp = om.getSourceOp(inputTensor);

        for (auto& out : prevOp->getOutputTensor())
        {
            if(alreadyInVec(sliceInputs, out))
                continue;

            auto children = findSinkLayers(dm, out);

            // Apply transformation only if all children are Slice Ops
            if (std::all_of(children.cbegin(), children.cend(),
                [](const mv::Data::OpListIterator& op) { return op->getOpType() == "Slice"; }))
            {
                sliceInputs.push_back(out);
            }
        }
    }

    for (const auto& tensor : sliceInputs)
    {
        moveFQBeforeSlice(dm, tensor);
    }
}
