#include "math.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

static void streamingTilingFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& target,
                                        mv::Element& passDesc,
                                        mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(streamingTiling)
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

    Tiling() :start_({0,0,0,0}),size_({0,0,0,0}),axis_(""),childTiles_(0) {};
    Tiling( mv::Shape& start,mv::Shape& size)
            : start_(start),size_(size),axis_(""),childTiles_(0)
    {
    };

    Tiling( std::string& axis,std::size_t tiles)
            : start_({0,0,0,0}),size_({0,0,0,0}),axis_(axis),childTiles_(tiles)
    {

    }
    Tiling( mv::Shape& start,mv::Shape& size, std::string axis,std::size_t childTiles)
            : start_(start),size_(size),axis_(axis),childTiles_(childTiles)
    {
    }

    Tiling& operator=(const Tiling& other)
    {
        start_=other.start_;
        size_ =other.size_;
        axis_ =other.axis_;
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
    void setChildTile( Tiling& tile,unsigned index ) { childTiles_[index] = tile; };

    void resizeNumberOfTiles(std::size_t children) { childTiles_.resize(children); } ;

    //TODO::build proper stream out of this
    void printOut(unsigned depth) const
    {
        for(unsigned tab =0; tab<depth; tab++)
            std::cout<<"\t";
        std::cout << "Master : " << size_.toString()  << std::endl;

        for(unsigned tab =0; tab<depth; tab++)
            std::cout<<"\t";
        for( auto& tile : childTiles_)
        {
            std::cout << "\tChild: ";
            tile.printOut(depth+1);\
        }
    };
};

mv::Data::TensorIterator solveChannelTiling(mv::OpModel& om,mv::Data::OpListIterator op,Tiling& tiling);
mv::Data::TensorIterator solveSpatialTiling(mv::OpModel& om,mv::Data::OpListIterator op,Tiling& tiling);

std::function<mv::Data::TensorIterator(mv::OpModel&,mv::Data::OpListIterator,Tiling&)> convSpatialTiling = solveSpatialTiling;
std::function<mv::Data::TensorIterator(mv::OpModel&,mv::Data::OpListIterator,Tiling&)> convOutChannelTiling = solveChannelTiling;

std::map<std::string, std::function<mv::Data::TensorIterator(mv::OpModel&,mv::Data::OpListIterator,Tiling&)>> streamSplit =
{
    {"W",solveSpatialTiling},
    {"H",solveSpatialTiling},
    {"C",solveChannelTiling} //TBD: for other operations that conv.
};


mv::Data::TensorIterator solveChannelTiling(mv::OpModel& om,mv::Data::OpListIterator op,Tiling& tiling)
{
    //TODO:: write the function
    return op->getOutputTensor(0);
}

mv::Data::TensorIterator solveSpatialTiling(mv::OpModel& om,mv::Data::OpListIterator op,Tiling& tiling)
{
    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto kernelTensor = op->getInputTensor(1);
    auto outputTensor = op->getOutputTensor(0);

    auto number_of_spltis = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    std::vector<mv::Shape> spatial_indexes(number_of_spltis);
    std::vector<mv::Data::TensorIterator> slices(number_of_spltis);
    std::vector<mv::Data::TensorIterator> convs(number_of_spltis);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_spltis);

    auto kernelStride = op->get<std::array<unsigned short, 2>>("stride");

    // pad_left, pad_right, pad_top, pad_bottom
    //TODO: cleaner solution for paddings..
    //assuming order of paddings: left,right,top,bottom
    auto padding = op->get<std::array<unsigned short, 4>>("padding");
    auto startPad = padding;
    auto endPad = padding;
    auto currentPad = padding;

    std::array<unsigned short,4> middle_pad = {0,0,0,0};

    if(axisToSplit == mv::Shape::getAxis("W"))
    {
        startPad[1] = 0;
        endPad[0] = 0;
    }
    else if(axisToSplit == mv::Shape::getAxis("H"))
    {
        startPad[3] = 0;
        endPad[2] = 0;
    }

    for (unsigned split = 0; split < number_of_spltis; split++)
    {

        auto slice = om.slice(inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize(),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_slice_" + std::to_string(split));

        if(split == 0)
            currentPad = startPad;
        else if(split == (number_of_spltis -1))
            currentPad = endPad;
        else
            currentPad = middle_pad;

        auto conv = om.conv(slice,
                                kernelTensor,
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"),
                                op->get<mv::QuantizationParams>("quantParams"),
                                op->getName() + "_split_" + std::to_string(split));

        auto newOp = om.getSourceOp(conv);

        newOp->set<bool>("splitted",true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->set<unsigned>("opId",op->get<unsigned>("opId"));

        slices[split] = slice;
        convs[split] = conv;

//        om.defineFlow(kernelTensor,newOp,1); //TODO:: review.
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    for(unsigned split = 0 ; split < number_of_spltis; split++)
    {
        mv::Tensor::MemoryLocation inputLocation;
        mv::Tensor::MemoryLocation outputLocation;
        if(childTiles[split].childTiles().size() > 1)
        {
            //has chidren. Inherit
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
        slices[split]->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
        convs[split]->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
    }


    for(unsigned split = 0; split < number_of_spltis; split++)
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
                    tiling.getAxis(),
                    op->get<mv::QuantizationParams>("quantParams"),
                    op->getName() + "_concat_");

    concat->set<mv::Tensor::MemoryLocation>("Location",outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

void generateSpatialTiling(mv::Data::OpListIterator op,Tiling& tiling)
{
    auto numberOfSplits = tiling.childTiles().size();
    auto weightTensor = op->getInputTensor(1);

    auto inputShape = tiling.getSize();
    auto weightsShape = weightTensor->getShape();

    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    int newSize = ceil( ((double)inputShape[axisToSplit]) / ((double)numberOfSplits));
    int remainderSize = inputShape[axisToSplit] - (newSize*(numberOfSplits -1));

    //todo:: check for original weights not the aligned one
    auto kernelDin = ( axisToSplit == mv::Shape::getAxis("W")) ? mv::KERNEL_WIDTH : mv::KERNEL_HEIGHT;
    auto kernelSize = weightsShape[kernelDin];
    unsigned kernelStep = kernelSize / 2; //TODO:: Check with HW and also with Dilation

    //todo:: is there any macro for kernel w/h order?
    auto kernelAxis = (axisToSplit == mv::Shape::getAxis("W")) ? 0 : 1;
    auto kernelStride = op->get<std::array<unsigned short, 2>>("stride")[kernelAxis];

    unsigned startCoord = 0;

    for(auto split = 0; split < numberOfSplits; split++)
    {
        mv::Shape tileStart({0,0,0,0});
        mv::Shape tileSize = inputShape;

        tileStart[axisToSplit] = startCoord;

        if(split == 0)
            startCoord += (newSize - kernelStep);
        else
            startCoord += kernelStep;

        if(split == 0)
            tileSize[axisToSplit] = (newSize + kernelStep);
        else if( split == (numberOfSplits-1) )
            tileSize[axisToSplit] = (remainderSize + kernelStep);
        else
            tileSize[axisToSplit] = newSize + (2 * kernelStep);

        Tiling newTile(tileStart,tileSize);
        tiling.setChildTile(newTile,split);
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

    for(auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
//        std::cout<< "splitDoing " << opIt->getName() << std::endl;
        std::string opType = opIt->getOpType();
        if( opType=="Conv" && !opIt->hasAttr("splitted"))
        {
            //TODO:: get this as param or something!
            //the startingTile is the "big tensor". (currently any conv will be split based on one JSON specifier)
            //###################################################################################################
            //currently we will drive the schedule by the output tensor....
            //TODO:: check with POC if the schedule accounts for the overlaps and inputStrides
            //TODO:: also consider dilation factor

            int numberOfSplits = passDesc.get<int>("numberOfSplits");

            mv::Shape startingCorner( {0,0,0,0});
            auto masterSize = opIt->getInputTensor(0)->getShape();

            Tiling masterTile(startingCorner,masterSize,"H",numberOfSplits);
            generateSpatialTiling(opIt,masterTile);

            for( auto& tile : masterTile.childTiles())
            {
                tile.setAxis("W");
                tile.resizeNumberOfTiles(numberOfSplits);
                generateSpatialTiling(opIt,tile);
            }

            auto sourceTensor = opIt->getInputTensor(0);
            auto parentOpIt = om.getSourceOp(sourceTensor);

            //######################################################################################################


            // auto result = solveSpatialTiling(om,opIt,masterTile);
            auto result = (streamSplit[masterTile.getAxis()])(om,opIt,masterTile);

            // Important: do not change the order of this ops
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
                opsToLink[j]->setInputTensor(result, inputSlots[j]);
                om.defineFlow(result, opsToLink[j], inputSlots[j]);
            }

            //TODO:: If the OpIt is set to the parentOpIt the iteration will  start from the beginning, and will
            // also include the "newly splitted" layers. But we deleted the original op, so we can't increment that.
            // is there a way to go to the "original next op" ? Currently there is a trait called "splitted" added,
            // and we check for that, so we do not split again.....
            opIt = parentOpIt;
//            std::cout<< "nextOneShouldBe" << opIt->getName() << std::endl;
        }
    }
}
