#include "math.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"

static void streamingTilingFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& target,
                                        mv::Element& passDesc,
                                        mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StreamingTiling)
                .setFunc(streamingTilingFcn)
                .setDescription(
                        "splits only over H for DDR streaming");
    }
}

class Tiling {
private:
    mv::Shape start_; //todo:: use shape!?
    mv::Shape size_;

    std::string axis_;
    std::vector<Tiling> childTiles_;

public:

    Tiling() :start_({0,0,0,0}), size_({0,0,0,0}), axis_(""), childTiles_(0) {};
    Tiling( mv::Shape& start, mv::Shape& size)
            : start_(start), size_(size), axis_(""), childTiles_(0)
    {
    };

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

    std::string& getAxis() { return axis_; };
    void setAxis(const std::string axis) { axis_ = axis; };

    mv::Shape& getStartCoord() { return start_; };
    void setStartCoord(mv::Shape start) { start_ = start; };

    mv::Shape& getSize() { return size_; };
    void setSize(mv::Shape size) { size_ = size; };

    std::vector<Tiling>& childTiles() { return childTiles_; };
    void setChildTile(Tiling& tile, unsigned index) { childTiles_[index] = tile; };

    void resizeNumberOfTiles(std::size_t children) { childTiles_.resize(children); } ;

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
    };
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

static void setStreamingStrategy(const mv::pass::PassEntry& pass, mv::ComputationModel& model,std::map<std::string, std::vector<opStreamingSplitDef>>& thisGraphStrategy)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // get ops to split and number of splits from descriptor
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        std::cout << "SET STREAMING STRATEGY EXITING: no strategy defined in JSON" << std::endl;
        pass.log(mv::Logger::MessageType::Info, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    // each s refers to the name of an op, from the JSON strategy list
    for (auto s: strategyList)
    {
        std::vector<opStreamingSplitDef> opxSplits;

        std::string nodeName = s.get<std::string>("name_filter") ;
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i=0; i<splitList.size(); i++)
        {
            opStreamingSplitDef opxSplitx;
            if (splitList[i].hasAttr("H"))
            {
                opxSplitx.axis= "H";
                opxSplitx.numSplits= splitList[i].get<int>("H");
                opxSplits.push_back(opxSplitx);
            }
            else if (splitList[i].hasAttr("W"))
            {
                opxSplitx.axis= "W";
                opxSplitx.numSplits= splitList[i].get<int>("W");
                opxSplits.push_back(opxSplitx);
            }
            else if (splitList[i].hasAttr("K"))
            {
                opxSplitx.axis= "K";
                opxSplitx.numSplits= splitList[i].get<int>("K");
                opxSplits.push_back(opxSplitx);
            }
        }
        thisGraphStrategy.insert(std::pair<std::string, std::vector<opStreamingSplitDef>>(nodeName,opxSplits));
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
    std::cout<< "  In solveWeightsTiling " << std::endl ;

    mv::OpModel om(model);
    mv::DataModel dm(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto kernelTensor = op->getInputTensor(1);
    auto outputTensor = op->getOutputTensor(0);

    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);

    auto kernelStride = op->get<std::array<unsigned short, 2>>("stride");

    size_t biasStartIndex = 0;
    size_t biasEndIndex = 0;
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

        auto conv = om.conv(inputTensor,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"),
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

        slices[split] = slice;
        convs[split] = conv;

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
            inputLocation.relocate(mv::Tensor::MemoryLocation::CMX);
            outputLocation.relocate(mv::Tensor::MemoryLocation::CMX);
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
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);

    concat->set<mv::Tensor::MemoryLocation>("Location",outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, Tiling& tiling)
{
    mv::OpModel om(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto outputTensor = op->getOutputTensor(0);
    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    std::vector<mv::Shape> spatial_indexes(number_of_splits);
    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);

    auto kernelStride = op->get<std::array<unsigned short, 2>>("stride");

    // pad_left, pad_right, pad_top, pad_bottom
    //TODO: cleaner solution for paddings..
    //assuming order of paddings: left,right,top,bottom
    auto padding = op->get<std::array<unsigned short, 4>>("padding");
    auto startPad = padding;
    auto endPad = padding;
    auto currentPad = padding;

    std::array<unsigned short, 4> middle_pad = {0,0,0,0};

    if (axisToSplit == mv::Shape::getAxis("W"))
    {
        startPad[1] = 0;
        endPad[0] = 0;
    }
    else if (axisToSplit == mv::Shape::getAxis("H"))
    {
        startPad[3] = 0;
        endPad[2] = 0;
    }

    for (unsigned split = 0; split < number_of_splits; split++)
    {

        auto slice = om.slice(inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_slice_" + std::to_string(split));
        om.getSourceOp(slice)->set<unsigned>("opId", opId);

        if (split == 0)
            currentPad = startPad;
        else if (split == (number_of_splits -1))
            currentPad = endPad;
        else
            currentPad = middle_pad;

        mv::Data::TensorIterator newTensor;
        std::string opType = op->getOpType();
        if (opType=="MaxPool")
        {
            newTensor = om.maxPool(slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                currentPad,
                                op->get<const bool>("exclude_pad"),
                                op->get<std::string>("auto_pad"),
                                op->get<std::string>("rounding_type"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_split_" + std::to_string(split));
        }
        else
        {
            newTensor = om.conv(slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_split_" + std::to_string(split));
        }

        auto newOp = om.getSourceOp(newTensor);

        if (op->hasAttr("bias"))
        {
            auto biasTensorName = op->get<std::string>("bias");
            om.addAttr(newOp, "bias", biasTensorName);
        }

        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->set<unsigned>("opId", opId);

        slices[split] = slice;
        convs[split] = newTensor;

//        om.defineFlow(kernelTensor,newOp,1); //TODO:: review.
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    for (unsigned split = 0 ; split < number_of_splits; split++)
    {
        mv::Tensor::MemoryLocation inputLocation;
        mv::Tensor::MemoryLocation outputLocation;
        if (childTiles[split].childTiles().size() > 1)
        {
            //has chidren. Inherit
            inputLocation.relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

            // std::cout << "More children. Inheriting " << slices[split]->getName() << " to " << inputLocation.toString() << " from " << inputTensor->getName() <<  std::endl;
            // std::cout << "More children. Inheriting " << convs[split]->getName() << " to " << outputLocation.toString() << " from " << outputTensor->getName() <<  std::endl;
        }
        else
        {
            //no more children.
            //todo:: Expose in JSON config the "Default stream location"
            inputLocation.relocate(mv::Tensor::MemoryLocation::CMX);
            outputLocation.relocate(mv::Tensor::MemoryLocation::CMX);
            inputLocation.force();
            outputLocation.force();

            // std::cout << "No more children deciding " << slices[split]->getName() << " to " << inputLocation.toString() << std::endl;
            // std::cout << "No more children deciding " << convs[split]->getName() << " to " << outputLocation.toString() << std::endl;
        }
        slices[split]->set<mv::Tensor::MemoryLocation>("Location", inputLocation);
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

    auto concat = om.concat(final_outputs,
                    tiling.getAxis(),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "concat_");
    om.getSourceOp(concat)->set<unsigned>("opId", opId);

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

void generateSpatialTiling(mv::Data::OpListIterator op,Tiling& tiling, std::vector<opStreamingSplitDef> opStrategy, int nesting)
{
    std::cout<< "  In generateSpatialTiling, op " << op->getName() << " nesting = " << nesting ;
    auto numberOfSplits = tiling.childTiles().size();
    std::cout<< " numsplits = " << numberOfSplits << std::endl ;

    auto inputShape = tiling.getSize();

    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());

//    int newOutputSize = ceil( ((double)inputShape[axisToSplit]) / ((double)numberOfSplits));
//    int remainderSize = inputShape[axisToSplit] - (newSize * (numberOfSplits -1));

//    int newOutputSize =  (double) inferOutputSize(inputShape[axisToSplit])

    //todo:: check for original weights not the aligned one
    size_t kernelSize;
    std::string opType = op->getOpType();
    if (opType == "Conv")
    {
        auto weightTensor = op->getInputTensor(1);
        auto weightsShape = weightTensor->getShape();
        auto kernelDin = (axisToSplit == mv::Shape::getAxis("W")) ? mv::KERNEL_WIDTH : mv::KERNEL_HEIGHT;
        kernelSize = weightsShape[kernelDin];
    }
    else
    {
        kernelSize = op->get<std::array<unsigned short, 2UL>>("kSize")[0];
    }
    unsigned kernelStep = kernelSize / 2; //TODO:: Check with HW and also with Dilation

    //todo:: is there any macro for kernel w/h order?
    auto kernelAxis = (axisToSplit == mv::Shape::getAxis("W")) ? 0 : 1;
    auto kernelStride = op->get<std::array<unsigned short, 2>>("stride")[kernelAxis];

    auto padding = op->get<std::array<unsigned short, 4>>("padding");
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
    int newOutputSize = ceil( (double)(outputSize) / (double)numberOfSplits);
    int remainderOutputSize = outputSize - ( newOutputSize *(numberOfSplits -1));

    unsigned startCoord = 0;
    for (auto split = 0; split < numberOfSplits; split++)
    {
        mv::Shape tileStart({0,0,0,0});
        mv::Shape tileSize = inputShape;

        tileStart[axisToSplit] = startCoord;

        if (split == 0)
            tileSize[axisToSplit] = inferInputSize(newOutputSize,padStart,0,kernelSize,kernelStride);
        else if (split == (numberOfSplits-1))
            tileSize[axisToSplit] = inferInputSize(remainderOutputSize,0,padEnd,kernelSize,kernelStride);
        else
            tileSize[axisToSplit] = inferInputSize(remainderOutputSize,0,0,kernelSize,kernelStride);

        if (split == 0)
            startCoord += tileSize[axisToSplit] - (inferInputSize(newOutputSize,0,0,kernelSize,kernelStride) - tileSize[axisToSplit]);
        else
            startCoord += tileSize[axisToSplit];

        Tiling newTile(tileStart, tileSize);
        tiling.setChildTile(newTile, split);
    }

    nesting++;
    if (nesting<opStrategy.size() )
    {
        for( auto& tile : tiling.childTiles())
        {
            tile.setAxis( opStrategy[nesting].axis );
            tile.resizeNumberOfTiles(opStrategy[nesting].numSplits) ;
            generateSpatialTiling(op,tile,opStrategy,nesting);
        }
    }
}

void generateWeightsTiling(mv::Data::OpListIterator op,Tiling& tiling, std::vector<opStreamingSplitDef> opStrategy, int nesting)
{
    std::cout<< "  In generateWeightsTiling, op " << op->getName() << " nesting = " << nesting ;
    auto numberOfSplits = tiling.childTiles().size();
    std::cout<< " numsplits = " << numberOfSplits << std::endl ;

    auto parentTileShape = tiling.getSize();

    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    int newSize = ceil( ((double)parentTileShape[axisToSplit]) / ((double)numberOfSplits));
    int remainderSize = parentTileShape[axisToSplit] - (newSize*(numberOfSplits -1));

    unsigned startCoord = 0;

    for(auto split = 0; split < numberOfSplits; split++)
    {
        mv::Shape tileStart({0,0,0,0});
        mv::Shape tileSize = parentTileShape;

        tileStart[axisToSplit] = startCoord;

        startCoord += newSize;

        if( split == (numberOfSplits-1) )
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
                                mv::TargetDescriptor& target,
                                mv::Element& passDesc,
                                mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    std::map<std::string, std::vector<opStreamingSplitDef>> thisGraphStrategy;
    setStreamingStrategy(pass, model, thisGraphStrategy);
    std::vector<opStreamingSplitDef> thisOpStrategy;

    std::cout<< "STREAMING PASS: entered" << std::endl ;

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
        std::cout<< "  checking " << masterOpName << std::endl;
        bool opHasSplittingStrategy = false;
        if (thisGraphStrategy.count(masterOpName)<1)
        {
            std::cout<< "  no streaming strategy for " << masterOpName << std::endl;
        }
        else
        {
            thisOpStrategy = thisGraphStrategy[masterOpName];
            std::cout<< "  streaming nesting depth is " << thisOpStrategy.size() << std::endl;
            opHasSplittingStrategy = true;
        }

        std::string opType = opIt->getOpType();
        if ((opType == "Conv" || (opType == "MaxPool")) && !opIt->hasAttr("splitted") && opHasSplittingStrategy)
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

            std::cout<< "   connecting "<< result->getName() <<" to " << opsToLink[0]->getName() << " input slot " <<  inputSlots[0] << std::endl ;
            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                opsToLink[j]->setInputTensor(result, inputSlots[j]);
                om.defineFlow(result, opsToLink[j], inputSlots[j]);
            }

        }
    }
    std::cout<< "STREAMING PASS: exit" << std::endl ;
}
