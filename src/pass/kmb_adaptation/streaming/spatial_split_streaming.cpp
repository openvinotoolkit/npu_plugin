#include "math.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"

static void streamingTilingFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry&,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StreamingTiling)
                .setFunc(streamingTilingFcn)
                .setDescription(
                        "splits only over H for DDR streaming");

        MV_REGISTER_PASS(StreamBinaryDataWeights)
        .setFunc(streamBinaryDataWeightsFcn)
        .setDescription(
            "stream the binary data of weights"
        );
    }
}

mv::Data::OpListIterator operationsReplacement(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    while(opIt.parentsSize() > 1)
    {
        auto paramOp = opIt.leftmostParent();
        ++paramOp;
        om.removeOp(paramOp);
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        //no need to trigger a cascade, we know what we are doing
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

class Tiling {
private:
    mv::Shape start_; //todo:: use shape!?
    mv::Shape size_;

    std::string axis_;
    std::vector<Tiling> childTiles_;

public:

    Tiling() :start_({0,0,0,0}), size_({0,0,0,0}), axis_(""), childTiles_(0) {}
    Tiling( mv::Shape& start, mv::Shape& size)
            : start_(start), size_(size), axis_(""), childTiles_(0)
    {
    }

    Tiling( std::string& axis, std::size_t tiles)
            : start_({0,0,0,0}), size_({0,0,0,0}), axis_(axis), childTiles_(tiles)
    {

    }
    Tiling( mv::Shape& start, mv::Shape& size, std::string axis, std::size_t childTiles)
            : start_(start), size_(size), axis_(axis), childTiles_(childTiles)
    {
    }

    Tiling& operator=(const Tiling& other)
    {
        start_= other.start_;
        size_ = other.size_;
        axis_ = other.axis_;
        childTiles_ = other.childTiles_;
        return *this;
    }

    std::string& getAxis() { return axis_; }
    void setAxis(const std::string axis) { axis_ = axis; }

    mv::Shape& getStartCoord() { return start_; }
    void setStartCoord(mv::Shape start) { start_ = start; }

    mv::Shape& getSize() { return size_; }
    void setSize(mv::Shape size) { size_ = size; }

    std::vector<Tiling>& childTiles() { return childTiles_; }
    void setChildTile(Tiling& tile, unsigned index) { childTiles_[index] = tile; }

    void resizeNumberOfTiles(std::size_t children) { childTiles_.resize(children); }

    //TODO::build proper stream out of this
    void printOut(unsigned depth) const
    {
        for (unsigned tab = 0; tab < depth; tab++)
            std::cout<<"\t";
        std::cout << "Master : " << size_.toString()  << std::endl;

        for (unsigned tab = 0; tab < depth; tab++)
            std::cout<<"\t";
        for (auto& tile : childTiles_)
        {
            std::cout << "\tChild: ";
            tile.printOut(depth+1);\
        }
    }
};

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, Tiling& tiling);
mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, Tiling& tiling);

std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, Tiling&)> convSpatialTiling = solveSpatialTiling;
std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, Tiling&)> convOutChannelTiling = solveWeightsTiling;

std::map<std::string, std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, Tiling&)>> streamSplit =
{
    {"W",solveSpatialTiling},
    {"H",solveSpatialTiling},
    {"K",solveWeightsTiling} //TBD: for other operations that conv.
};

struct opStreamingSplitDef
{
    std::string axis ;
    size_t numSplits ;
};

static void setStreamingStrategy(const mv::pass::PassEntry &pass, mv::ComputationModel &model, std::map<std::string, std::vector<opStreamingSplitDef>> &thisGraphStrategy)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // get ops to split and number of splits from descriptor
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        std::cout << "No strategy defined in JSON" << std::endl;
        pass.log(mv::Logger::MessageType::Info, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    // each s refers to the name of an op, from the JSON strategy list
    for (auto s : strategyList)
    {
        std::vector<opStreamingSplitDef> opxSplits;
        bool nodeHasSplit = false;
        std::string nodeName = s.get<std::string>("name_filter");
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (std::size_t i=0; i<splitList.size(); i++)
        {
            opStreamingSplitDef opxSplitx;
            if (splitList[i].hasAttr("H"))
            {
                if (splitList[i].get<int>("H")>1)
                {
                    opxSplitx.axis = "H";
                    opxSplitx.numSplits = splitList[i].get<int>("H");
//                    std::cout << "IN STREAMING PASS : H = " << opxSplitx.numSplits << std::endl ;
//                    if (opxSplitx.numSplits == (size_t) 3)
//                    {
//                        std::cout << "IN STREAMING PASS : H==3 detected" << std::endl ;
//                        opxSplitx.numSplits = 4;
//                    }
                    opxSplits.push_back(opxSplitx);
                    nodeHasSplit=true;
                    std::cout << "Streaming for node: " << nodeName << " has stream H = " << opxSplitx.numSplits << std::endl ;
                }
            }
            else if (splitList[i].hasAttr("W"))
            {
                if (splitList[i].get<int>("W")>1)
                {
                    opxSplitx.axis = "W";
                    opxSplitx.numSplits = splitList[i].get<int>("W");
                    opxSplits.push_back(opxSplitx);
                    nodeHasSplit=true;
                }
            }
            else if (splitList[i].hasAttr("C"))
            {
                if (splitList[i].get<int>("C")>1)
                {
                    opxSplitx.axis = "C";
                    opxSplitx.numSplits = splitList[i].get<int>("C");
                    opxSplits.push_back(opxSplitx);
                    nodeHasSplit=true;
                }
            }
            else if (splitList[i].hasAttr("K"))
            {
                if (splitList[i].get<int>("K")>1)
                {
                    std::cout << "Streaming for node: " << nodeName << " has stream K = " << splitList[i].get<int>("K") << std::endl ;
                    opxSplitx.axis = "K";
                    opxSplitx.numSplits = splitList[i].get<int>("K");
                    opxSplits.push_back(opxSplitx);
                    nodeHasSplit=true;
                }
            }
        }
        if (nodeHasSplit) thisGraphStrategy.insert(std::pair<std::string, std::vector<opStreamingSplitDef>>(nodeName, opxSplits));
    }
    /*
    std::vector<opStreamingSplitDef> op1Splits;
    std::vector<opStreamingSplitDef> op2Splits;
    opStreamingSplitDef opxSplitx;
    opxSplitx.axis= "H" ;
    opxSplitx.numSplits = 2 ;
    op1Splits.push_back(opxSplitx);
    opxSplitx.axis = "W" ;
    opxSplitx.numSplits = 4 ;
    op1Splits.push_back(opxSplitx);
    opxSplitx.axis = "H" ;
    opxSplitx.numSplits = 2 ;
    op1Splits.push_back(opxSplitx);
    opxSplitx.axis = "H" ;
    opxSplitx.numSplits = 2 ;
    op2Splits.push_back(opxSplitx);
    opxSplitx.axis = "W" ;
    opxSplitx.numSplits = 2 ;
    op2Splits.push_back(opxSplitx);

    thisGraphStrategy.insert(std::pair<std::string, std::vector<opStreamingSplitDef>>("conv0_cmx_",op1Splits));
    thisGraphStrategy.insert(std::pair<std::string, std::vector<opStreamingSplitDef>>("conv1_cmx_",op2Splits));
*/
}

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op,Tiling& tiling)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto inputTensor2Conv = inputTensor ;
    auto kernelTensor = op->getInputTensor(1);
    auto outputTensor = op->getOutputTensor(0);

    mv::QuantizationParams quantParams = {{},{},{},{}};
    if(inputTensor->hasAttr("quantParams"))
        quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");

    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);

    size_t biasStartIndex = 0;
    size_t biasEndIndex = 0;

    std::string splitStrategy = op->get<std::string>("splitStrategy");

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator slice;
        if (kernelTensor->hasAttr("quantParams"))
        {
            slice = om.slice(kernelTensor,
                            childTiles[split].getStartCoord(),
                            childTiles[split].getSize(),
                            kernelTensor->get<mv::QuantizationParams>("quantParams"),
                            kernelTensor->getName() + "_slice_" + std::to_string(split));
        }
        else
        {
            slice = om.slice(kernelTensor,
                            childTiles[split].getStartCoord(),
                            childTiles[split].getSize(),
                            {{}, {}, {}, {}},
                            kernelTensor->getName() + "_slice_" + std::to_string(split));
        }
        om.getSourceOp(slice)->set<unsigned>("opId", opId);

        auto conv = om.conv(inputTensor2Conv,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_split_" + std::to_string(split));
        if (op->hasAttr("bias"))
        {
            auto tileSize = childTiles[split].getSize()[axisToSplit];
            biasStartIndex = biasEndIndex;
            biasEndIndex = biasStartIndex + tileSize;

            auto biasTensorName = op->get<std::string>("bias");
            auto originalBiasTensor = dm.getTensor(biasTensorName);
            auto oiginalBiasData = originalBiasTensor->getData();
            if ( biasEndIndex > oiginalBiasData.size())
            {
                biasEndIndex = oiginalBiasData.size();
            }
            std::vector<mv::DataElement>::const_iterator biasFirst = oiginalBiasData.begin() + biasStartIndex;
            std::vector<mv::DataElement>::const_iterator biasLast = oiginalBiasData.begin() + biasEndIndex;
            std::vector<mv::DataElement> subBiasData(biasFirst, biasLast);

            std::string newBiasTensorName = mv::createBiasName(op->getName() + "_split_" + std::to_string(split));
            mv::Data::TensorIterator biasTensor;

            mv::Data::TensorIterator biasTensorX;
            if (originalBiasTensor->hasAttr("quantParams"))
            {
                auto biasAttrQPs = originalBiasTensor->get("quantParams");
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData, biasAttrQPs ));
            }
            else
            {
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData));
            }

            om.addAttr(om.getSourceOp(conv), "bias", biasTensorX->getName());
        }

        auto newOp = om.getSourceOp(conv);

        newOp->set<bool>("splitted",true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->set<unsigned>("opId",opId);
        newOp->set<std::string>("splitStrategy", splitStrategy);
        newOp->set<bool>("inputActivationSparsity", op->get<bool>("inputActivationSparsity"));
        newOp->set<bool>("outputActivationSparsity", op->get<bool>("outputActivationSparsity"));
        newOp->set<bool>("weightsSparsity", op->get<bool>("weightsSparsity"));

        slices[split] = slice;
        convs[split] = conv;

        bool enableSerialStreaming = true;
        if ((split>0)&&(enableSerialStreaming))
            cm.defineFlow(om.getSourceOp(convs[split-1]), om.getSourceOp(convs[split]));
    }

    kernelTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    for(unsigned split = 0 ; split < number_of_splits; split++)
    {
        mv::Tensor::MemoryLocation inputLocation;
        mv::Tensor::MemoryLocation outputLocation;
        if(childTiles[split].childTiles().size() > 1)
        {
            //has children. Inherit
            inputLocation.relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

            // std::cout << "More children. Inheriting " << slices[split]->getName() << " to " << inputLocation.print() << " from " << inputTensor->getName() <<  std::endl;
            // std::cout << "More children. Inheriting " << convs[split]->getName() << " to " << outputLocation.print() << " from " << outputTensor->getName() <<  std::endl;
        }
        else
        {
            //no more children.
            //todo:: Expose in JSON config the "Default stream location"
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            inputLocation.force();
            outputLocation.force();

            // std::cout << "No more children deciding " << slices[split]->getName() << " to " << inputLocation.print() << std::endl;
            // std::cout << "No more children deciding " << convs[split]->getName() << " to " << outputLocation.print() << std::endl;
        }
        slices[split]->set<mv::Tensor::MemoryLocation>("Location", inputLocation);
        convs[split]->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
    }

    for(unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if(childTiles[split].childTiles().size() > 1)
        {
            // std::cout << "recurs doing " << convs[split]->getName() << std::endl;
            // out = solveSpatialTiling(om,om.getSourceOp(convs[split]),childTiles[split]);
            out = (streamSplit[childTiles[split].getAxis()])(om,om.getSourceOp(convs[split]),childTiles[split]);
            om.removeOp( om.getSourceOp(convs[split]));
        }
        else
        {
            out = convs[split];
        }
        final_outputs[split] = out;
    }

    auto concat = om.concat(final_outputs,
                    "C",
//                    tiling.getAxis(),
                    op->get<mv::DType>("dType"),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);

    concat->set<mv::Tensor::MemoryLocation>("Location",outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, Tiling& tiling)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto outputTensor = op->getOutputTensor(0);
    auto opId = op->get<unsigned>("opId");
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    std::vector<mv::Shape> spatial_indexes(number_of_splits);
    std::vector<std::vector<mv::Data::TensorIterator>> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    std::array<unsigned short, 2> kernelStride;
    if (op->hasAttr("stride"))
        kernelStride = op->get<std::array<unsigned short, 2>>("stride");
    else
        kernelStride = {1,1};//fake stride

    // pad_left, pad_right, pad_top, pad_bottom
    //TODO: cleaner solution for paddings..
    //assuming order of paddings: left,right,top,bottom
    std::array<unsigned short, 4> padding;
    if (op->hasAttr("padding"))
        padding = op->get<std::array<unsigned short, 4>>("padding");
    else
    {
        padding = {0, 0, 0, 0};
    }

    auto startPad = padding;
    auto endPad = padding;
    auto middlePad = padding;
    auto currentPad = padding;

    if (axisToSplit == mv::Shape::getAxis("W"))
    {
        startPad[1] = 0;
        endPad[0] = 0;
        middlePad[0] = 0;
        middlePad[1] = 0;
    }
    else if (axisToSplit == mv::Shape::getAxis("H"))
    {
        startPad[3] = 0;
        endPad[2] = 0;
        middlePad[2] = 0;
        middlePad[3] = 0;
    }

    for (unsigned split = 0; split < number_of_splits; split++)
    {

        if (split == 0)
            currentPad = startPad;
        else if (split == (number_of_splits -1))
            currentPad = endPad;
        else
            currentPad = middlePad;

        mv::Data::TensorIterator newTensor;
        std::string opType = op->getOpType();
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv")
        {
            auto inputTensor = op->getInputTensor(0);

            auto slice = om.slice(inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_slice_" + std::to_string(split));
            om.getSourceOp(slice)->set<unsigned>("opId", opId);
            if (opType == "MaxPool")
                newTensor = om.maxPool(slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                currentPad,
                                op->get<const bool>("exclude_pad"),
                                op->get<std::string>("auto_pad"),
                                op->get<std::string>("rounding_type"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_split_" + std::to_string(split));

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_split_" + std::to_string(split));

            if (opType == "Conv")
                newTensor = om.conv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"),
                                op->get<mv::DType>("dType"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_split_" + std::to_string(split));
            slices[split].push_back(slice);
        }
        else if (opType == "Add" || opType == "Subtract" || opType == "Multiply")
        {
            for (auto i = 0; i < 2; i++)
            {
                auto inputTensor = op->getInputTensor(i);

                auto slice = om.slice(inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                inputTensor->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_slice_" + std::to_string(split) + "_" + std::to_string(i));
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices[split].push_back(slice);
            }
            auto addFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const mv::DType& dType, const std::string& s){ return om.add(vec, dType, quantParams, s);};
            auto subFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const mv::DType& dType, const std::string& s){ return om.subtract(vec, dType, quantParams, s);};
            auto multFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const mv::DType& dType, const std::string& s){ return om.multiply(vec, dType, quantParams, s);};

            auto dpuTaskMap = std::map<std::string, std::function<mv::Data::TensorIterator (std::vector< mv::Data::TensorIterator >&, const mv::QuantizationParams&, const mv::DType&, const std::string&)>>
                                                    {{"Add", addFcn},
                                                    {"Subtract", subFcn},
                                                    {"Multiply", multFcn}};
            auto dpuElementWiseFunctor = (dpuTaskMap.at(opType));
            newTensor = dpuElementWiseFunctor(slices[split], op->get<mv::QuantizationParams>("quantParams"), op->get<mv::DType>("dType"), op->getName() + "_split_" + std::to_string(split));
        }
        else
        {
            throw mv::RuntimeError(om, opType + " not supported for streaming");
        }

        auto newOp = om.getSourceOp(newTensor);

        if (op->hasAttr("bias"))
        {
            auto biasTensorName = op->get<std::string>("bias");
            om.addAttr(newOp, "bias", biasTensorName);
        }

        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->set<unsigned>("opId", opId);
        newOp->set<std::string>("splitStrategy", splitStrategy);
        newOp->set<bool>("inputActivationSparsity", op->get<bool>("inputActivationSparsity"));
        newOp->set<bool>("outputActivationSparsity", op->get<bool>("outputActivationSparsity"));
        newOp->set<bool>("weightsSparsity", op->get<bool>("weightsSparsity"));
        if (op->hasAttr("postOpType"))
        {
            newOp->set<std::string>("postOpType", op->get<std::string>("postOpType"));
            if (newOp->get<std::string>("postOpType") == "LeakyRelu")
                newOp->set<double>("alpha", op->get<double>("alpha"));
        }
        else if (op->hasAttr("postOpTypes"))
        {
            newOp->set<std::vector<std::string>>("postOpTypes", op->get<std::vector<std::string>>("postOpTypes"));
            std::vector<std::string> postOpTypes = op->get<std::vector<std::string>>("postOpTypes");

            if (op->getOutputTensor(0)->getDType() ==  mv::DType("Float16"))
            {
                newOp->set<double>("minimum", op->get<double>("minimum"));
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Maximum") != postOpTypes.end())
                    newOp->set<double>("maximum", op->get<double>("maximum"));
            }
            else
            {
                newOp->set<int64_t>("minimum", op->get<int64_t>("minimum"));
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Maximum") != postOpTypes.end())
                    newOp->set<int64_t>("maximum", op->get<int64_t>("maximum"));
            }
        }

        convs[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(convs[split-1]), om.getSourceOp(convs[split]));
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    for (unsigned split = 0 ; split < number_of_splits; split++)
    {
        auto numInputs = slices[split].size();
        std::vector<mv::Tensor::MemoryLocation> inputLocation(numInputs);
        mv::Tensor::MemoryLocation outputLocation;
        if (childTiles[split].childTiles().size() > 1)
        {
            //has chidren. Inherit
            for (std::size_t i = 0; i < numInputs; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                inputLocation[i].relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            }
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

            // std::cout << "More children. Inheriting " << slices[split]->getName() << " to " << inputLocation.toString() << " from " << inputTensor->getName() <<  std::endl;
            // std::cout << "More children. Inheriting " << convs[split]->getName() << " to " << outputLocation.toString() << " from " << outputTensor->getName() <<  std::endl;
        }
        else
        {
            //no more children.
            //todo:: Expose in JSON config the "Default stream location"
            for (std::size_t i = 0; i < numInputs; i++)
            {
                inputLocation[i].relocate(mv::Tensor::MemoryLocation::NNCMX);
                inputLocation[i].force();
            }
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.force();

            // std::cout << "No more children deciding " << slices[split]->getName() << " to " << inputLocation.toString() << std::endl;
            // std::cout << "No more children deciding " << convs[split]->getName() << " to " << outputLocation.toString() << std::endl;
        }
        for (std::size_t i = 0; i < numInputs; i++)
            slices[split][i]->set<mv::Tensor::MemoryLocation>("Location", inputLocation[i]);
        convs[split]->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            // std::cout << "recurs doing " << convs[split]->getName() << std::endl;
            // out = solveSpatialTiling(om,om.getSourceOp(convs[split]),childTiles[split]);
            out = (streamSplit[childTiles[split].getAxis()])(om, om.getSourceOp(convs[split]), childTiles[split]);
            om.removeOp(om.getSourceOp(convs[split]));
        }
        else
        {
            out = convs[split];
        }
        final_outputs[split] = out;
    }
    //debug
    std::vector<mv::Shape> final_outputs_deb(number_of_splits);
    for (std::size_t i=0; i < number_of_splits; ++i)
        final_outputs_deb[i] = final_outputs[i]->getShape();

    auto concat = om.concat(final_outputs,
                    tiling.getAxis(),
                    op->get<mv::DType>("dType"),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);

    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

static inline int inferInputSize( int outputSize, int padding_start, int padding_end,int kernel_size,int kernel_stride)
{
    int inputSize =  ((outputSize -1) * kernel_stride)  -padding_start - padding_end + kernel_size;
    return inputSize;
}

static inline int inferOutputSize( int inputSize, int padding_start, int padding_end,int kernel_size, int kernel_stride)
{
    int outputSize = ( inputSize + padding_start + padding_end - kernel_size) / kernel_stride + 1;
    return outputSize;
}

void generateSpatialTiling(mv::Data::OpListIterator op,Tiling& tiling, std::vector<opStreamingSplitDef> opStrategy, unsigned int nesting)
{
    //std::cout<< "  In generateSpatialTiling, op " << op->getName() << " nesting = " << nesting ;
    auto numberOfSplits = tiling.childTiles().size();
    //std::cout<< " numsplits = " << numberOfSplits << std::endl ;

    auto inputShape = tiling.getSize();

    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());

    //todo:: check for original weights not the aligned one
    size_t kernelSize;
    std::string opType = op->getOpType();
    if (opType == "Conv" || opType == "DepthwiseConv")
    {
        auto weightTensor = op->getInputTensor(1);
        auto weightsShape = weightTensor->getShape();
        auto kernelDin = (axisToSplit == mv::Shape::getAxis("W")) ? mv::KERNEL_WIDTH : mv::KERNEL_HEIGHT;
        kernelSize = weightsShape[kernelDin];
    }
    else
    {
        if (op->hasAttr("kSize"))
        {
            kernelSize = op->get<std::array<unsigned short, 2UL>>("kSize")[0];
        }
        else
        {
            //static const std::array<unsigned short, 2> FAKE_KERNEL = {1,1};
            //static const std::array<unsigned short, 2> FAKE_STRIDE = {1,1};

            kernelSize = 1;//fake kernel
        }
    }

    //todo:: is there any macro for kernel w/h order?
    auto kernelAxis = (axisToSplit == mv::Shape::getAxis("W")) ? 0 : 1;
    unsigned short kernelStride;
    if (op->hasAttr("stride"))
    {
        kernelStride = op->get<std::array<unsigned short, 2>>("stride")[kernelAxis];
    }
    else
    {
        kernelStride = 1;//fake stride
    }
    std::array<unsigned short, 4> padding;
    if (op->hasAttr("padding"))
        padding = op->get<std::array<unsigned short, 4>>("padding");
    else
    {
        padding = {0,0,0,0};
    }

    int padStart,padEnd;

    if (axisToSplit == mv::Shape::getAxis("W"))
    {
        padStart = padding[0];
        padEnd = padding[1];
    }
    else if (axisToSplit == mv::Shape::getAxis("H"))
    {
        padStart = padding[2];
        padEnd = padding[3];
    }


    int outputSize =  inferOutputSize(inputShape[axisToSplit],padStart,padEnd,kernelSize,kernelStride);
    int newOutputSize = trunc( (double)(outputSize) / (double)numberOfSplits);
    int remainderOutputSize = outputSize - ( newOutputSize *(numberOfSplits -1));

    unsigned startCoord = 0;
    for (std::size_t split = 0; split < numberOfSplits; split++)
    {
        mv::Shape tileStart({0,0,0,0});
        mv::Shape tileSize = inputShape;

        tileStart[axisToSplit] = startCoord;

        if (split == 0)
            tileSize[axisToSplit] = inferInputSize(newOutputSize,padStart,0,kernelSize,kernelStride);
        else if (split == (numberOfSplits-1))
            tileSize[axisToSplit] = inferInputSize(remainderOutputSize,0,padEnd,kernelSize,kernelStride);
        else
            tileSize[axisToSplit] = inferInputSize(newOutputSize,0,0,kernelSize,kernelStride);

        Tiling newTile(tileStart, tileSize);
        tiling.setChildTile(newTile, split);

        // Compute start coordinates for the next tile
        // TODO: compute correct formula.
        if (split == 0)
            startCoord += newOutputSize * kernelStride - (inferInputSize(newOutputSize,0,0,kernelSize,kernelStride) - tileSize[axisToSplit]);
        else
            startCoord += newOutputSize * kernelStride;
    }

    nesting++;
    if (nesting < opStrategy.size())
    {
        for(auto& tile : tiling.childTiles())
        {
            tile.setAxis( opStrategy[nesting].axis );
            tile.resizeNumberOfTiles(opStrategy[nesting].numSplits) ;
            generateSpatialTiling(op,tile,opStrategy,nesting);
        }
    }
}

void generateWeightsTiling(mv::Data::OpListIterator op,Tiling& tiling, std::vector<opStreamingSplitDef> opStrategy, unsigned int nesting)
{
    auto numberOfSplits = tiling.childTiles().size();

    auto parentTileShape = tiling.getSize();

    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    int newSize = ceil( ((double)parentTileShape[axisToSplit]) / ((double)numberOfSplits));
    int remainderSize = parentTileShape[axisToSplit] - (newSize*(numberOfSplits -1));

    unsigned startCoord = 0;

    for(std::size_t split = 0; split < numberOfSplits; split++)
    {
        mv::Shape tileStart({0,0,0,0});
        mv::Shape tileSize = parentTileShape;

        tileStart[axisToSplit] = startCoord;

        startCoord += newSize;

        if(split == (numberOfSplits-1))
            tileSize[axisToSplit] = remainderSize;
        else
            tileSize[axisToSplit] = newSize;

        Tiling newTile(tileStart,tileSize);
        tiling.setChildTile(newTile,split);

    }

    nesting++;
    if (nesting<opStrategy.size() )
    {
        for( auto& tile : tiling.childTiles())
        {
            tile.setAxis( opStrategy[nesting].axis );
            tile.resizeNumberOfTiles(opStrategy[nesting].numSplits) ;
            generateWeightsTiling(op,tile,opStrategy,nesting);
        }

    }
}

void streamingTilingFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element& ,
                                mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    std::map<std::string, std::vector<opStreamingSplitDef>> thisGraphStrategy;
    setStreamingStrategy(pass, model, thisGraphStrategy);
    std::vector<opStreamingSplitDef> thisOpStrategy;

    //std::cout<< "STREAMING PASS: entered" << std::endl ;

    for (auto s: thisGraphStrategy)
    {
        std::string nodeName = s.first;
        if (!om.checkOp(nodeName))
        {
            pass.log(mv::Logger::MessageType::Error, nodeName + " is not present in model, skipping streaming");
            continue;
        }
        auto opIt =  om.getOp(nodeName);

        std::string masterOpName = opIt->getName();
        bool opHasSplittingStrategy = false;
        if (thisGraphStrategy.count(masterOpName)<1)
        {
            pass.log(mv::Logger::MessageType::Info, "  no streaming strategy for " + masterOpName);
        }
        else
        {
            thisOpStrategy = thisGraphStrategy[masterOpName];
            pass.log(mv::Logger::MessageType::Info, masterOpName + "  streaming nesting depth is " + std::to_string(thisOpStrategy.size()));
            opHasSplittingStrategy = true;
        }

        std::string opType = opIt->getOpType();
        bool isElementWise = (opType == "Add" || opType == "Subtract" || opType == "Multiply");

        if ((opType == "Conv" || opType == "DepthwiseConv" ||  (opType == "MaxPool") || isElementWise) && !opIt->hasAttr("splitted") && opHasSplittingStrategy)
        {
            //TODO:: get this as param or something!
            //the startingTile is the "big tensor". (currently any conv will be split based on one JSON specifier)
            //###################################################################################################
            //currently we will drive the schedule by the output tensor....
            //TODO:: check with POC if the schedule accounts for the overlaps and inputStrides
            //TODO:: also consider dilation factor

            int numberOfSplits = thisOpStrategy[0].numSplits ;
            std::string axisToSplit = thisOpStrategy[0].axis ;

            Tiling masterTile(axisToSplit, numberOfSplits);
            mv::Shape masterSize;
            if (axisToSplit == "K")
            {
                masterTile.setSize(opIt->getInputTensor(1)->getShape());
                generateWeightsTiling(opIt,masterTile,thisOpStrategy,0);
            }
            else
            {
                masterTile.setSize(opIt->getInputTensor(0)->getShape());
                generateSpatialTiling(opIt,masterTile,thisOpStrategy,0);
            }

            auto sourceTensor = opIt->getInputTensor(0);
            auto parentOpIt = om.getSourceOp(sourceTensor);

            //######################################################################################################

            auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile);

            // reconnect children to subgraph
            std::vector<mv::Data::OpListIterator> opsToLink;
            std::vector<std::size_t> inputSlots;
            for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                opsToLink.push_back(sinkFlow.sink());
                inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            }

            om.removeOp(opIt);

            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                //std::cout<< "   connecting "<< result->getName() <<" to " << opsToLink[j]->getName() << " input slot " <<  inputSlots[j] << std::endl ;

                opsToLink[j]->setInputTensor(result, inputSlots[j], false);
                om.defineFlow(result, opsToLink[j], inputSlots[j]);
            }

        }
        else
        {
            throw mv::RuntimeError(om, opType + " not supported for streaming");
        }

    }
    //std::cout<< "STREAMING PASS: exit" << std::endl ;
}


static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry& ,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& ,
                                        mv::Element& ,
                                        mv::Element &)
{
    //Need to duplicate the consts to number equal to streams, cause of the binary_data
    mv::OpModel om(model);

    std::set <std::string> removeConstantsSet;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();
        std::vector<mv::Data::TensorIterator> toSort;

        if (opType == "Slice" && opIterator->getInputTensor(0)->isPopulated())
        {
            auto inTensorSlice = opIterator->getInputTensor(0);
            removeConstantsSet.insert(om.getSourceOp(inTensorSlice)->getName());
            auto outTensorSlice = opIterator->getOutputTensor(0);
            auto parentOpIt = om.getSourceOp(opIterator->getInputTensor(0));
            mv::QuantizationParams tensorQuantizationParams = {{},{},{},{}};
            auto shape = outTensorSlice->getShape();
            if (outTensorSlice->isQuantized())
                tensorQuantizationParams = outTensorSlice->get<mv::QuantizationParams>("quantParams");

            auto newConstant = om.constantDataElement(outTensorSlice->getData(), shape,
                                                               outTensorSlice->getDType(), outTensorSlice->getOrder(),
                                                               tensorQuantizationParams, opIterator->getName() + "_weights");
            newConstant->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
            auto constantOp = om.getSourceOp(newConstant);
            if(opIterator->hasAttr("opId"))
            {
                unsigned currentOpId = opIterator->get<unsigned>("opId");
                constantOp->set<unsigned>("opId", currentOpId);
            }
            opIterator = operationsReplacement(parentOpIt, newConstant, om, opIterator);
        }
    }
    for (auto& opName:removeConstantsSet)
        om.removeOp(om.getOp(opName));
}
