#ifndef TILING_HPP_
#define TILING_HPP_

#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

#define TILE_DIM_W 0
#define TILE_DIM_H 1
#define TILE_DIM_C 2
#define TILE_DIM_K 3
#define TILE_DIM_N 4

namespace mv
{
    class Tiling {
    private:
        using tileShape = std::vector<std::size_t>;

        tileShape start_;
        tileShape size_;

        std::string axis_;
        std::vector<Tiling> childTiles_;

        void printShape(const tileShape& shape) const
        {
            std::cout<< "{";
            for(size_t i = 0; i < TILE_DIM_N; ++i)
                std::cout<< shape[i] << "," ;
            std::cout<<"}";
        }

    public:

        Tiling() :start_({0,0,0,0,0}), size_({0,0,0,0,0}), axis_(""), childTiles_(0) {}
        Tiling( Shape& actShape, Shape& kernelShape)
                : start_({0,0,0,0,0}), axis_(""), childTiles_(0)
        {
            size_.resize(5);
            size_[TILE_DIM_W] = actShape[mv::IO_WIDTH_DIMENSION];
            size_[TILE_DIM_H] = actShape[mv::IO_HEIGHT_DIMENSION];
            size_[TILE_DIM_C] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_K] = kernelShape[mv::KERNEL_OUTPUT_CHANNELS];
            size_[TILE_DIM_N] = actShape[mv::IO_BATCH_DIMENSION];
        }
        Tiling( Shape& actShape)
                : start_({0,0,0,0,0}),axis_(""), childTiles_(0)
        {
            size_.resize(5);
            size_[TILE_DIM_W] = actShape[mv::IO_WIDTH_DIMENSION];
            size_[TILE_DIM_H] = actShape[mv::IO_HEIGHT_DIMENSION];
            size_[TILE_DIM_C] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_K] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_N] = actShape[mv::IO_BATCH_DIMENSION];
        }

        Tiling(tileShape& start,tileShape& size) :
                start_(start),size_(size),axis_(""),childTiles_(0) {}

        Tiling( std::string& axis, std::size_t tiles)
                : start_({0,0,0,0,0}), size_({0,0,0,0,0}), axis_(axis), childTiles_(tiles)
        {

        }

        Tiling( Shape& start, Shape& size, std::string axis, std::size_t childTiles)
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

        tileShape& getStartCoord() { return start_; }
        void setStartCoord(tileShape start) { start_ = start; }

        tileShape& getSize() { return size_; }
        void setSize(tileShape size) { size_ = size; }

        std::vector<Tiling>& childTiles() { return childTiles_; }
        void setChildTile(Tiling& tile, unsigned index) { childTiles_[index] = tile; }

        void resizeNumberOfTiles(std::size_t children) { childTiles_.resize(children); }

        mv::Shape getActivationShape()
        {
            return mv::Shape({size_[TILE_DIM_W],size_[TILE_DIM_H],size_[TILE_DIM_C],size_[TILE_DIM_N]});
        }
        mv::Shape getActivationStart()
        {
            return mv::Shape({start_[TILE_DIM_W],start_[TILE_DIM_H],start_[TILE_DIM_C],start_[TILE_DIM_N]});
        }
        mv::Shape getKernelShape()
        {
            return mv::Shape({0,0,size_[TILE_DIM_C],size_[TILE_DIM_K]});
        }
        mv::Shape getKernelStart()
        {
            return mv::Shape({0,0,start_[TILE_DIM_C],start_[TILE_DIM_K]});
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

        void generateWeightsTiling()
        {
            auto numberOfSplits = childTiles_.size();
            auto parentTileShape = getSize();
            auto axisToSplit =  TILE_DIM_K; // the Size of the tile , is the size of the outputTensor... That is why we ask for "C" in the shape of the outTensor

            int newSize = ceil((double)(parentTileShape[axisToSplit] / numberOfSplits, 16));
            int remainderSize = parentTileShape[axisToSplit] - (newSize*(numberOfSplits -1));

            if(remainderSize == 0)
            {
                //this means that whoever gave the NR of streams did not take into account that channels need to be rounded
                numberOfSplits--;
                childTiles_.pop_back();
                remainderSize = newSize;
            }

            unsigned startCoord = 0;


            for(std::size_t split = 0; split < numberOfSplits; split++)
            {
                tileShape tileStart({0,0,0,0,0});
                tileShape tileSize = parentTileShape;

                tileStart[axisToSplit] = startCoord;
                startCoord += newSize;
                if(split == (numberOfSplits-1))
                    tileSize[axisToSplit] = remainderSize;
                else
                    tileSize[axisToSplit] = newSize;
                mv::Tiling newTile(tileStart,tileSize);
                setChildTile(newTile,split);
            }
        }

        void generateSpatialTiling(mv::Data::OpListIterator opIt)
        {
            auto numberOfSplits = childTiles().size();
            auto inputShape = getSize();
            auto axisToSplit =  mv::Shape::getAxis(getAxis());

            size_t kernelSize;
            std::string opType = opIt->getOpType();
            if (opType == "Conv" || opType == "DepthwiseConv")
            {
                auto weightTensor = opIt->getInputTensor(1);
                auto weightsShape = weightTensor->getShape();
                auto kernelDin = (axisToSplit == mv::Shape::getAxis("W")) ? mv::KERNEL_WIDTH : mv::KERNEL_HEIGHT;
                kernelSize = weightsShape[kernelDin];
            }
            else
            {
                if (opIt->hasAttr("kSize"))
                    kernelSize = opIt->get<std::array<unsigned short, 2UL>>("kSize")[0];
                else
                    kernelSize = 1;//fake kernel
            }

            //todo:: is there any macro for kernel w/h order?
            auto kernelAxis = (axisToSplit == mv::Shape::getAxis("W")) ? 0 : 1;
            unsigned short kernelStride;
            if (opIt->hasAttr("stride"))
                kernelStride = opIt->get<std::array<unsigned short, 2>>("stride")[kernelAxis];
            else
                kernelStride = 1;//fake stride
            std::array<unsigned short, 4> padding;
            if (opIt->hasAttr("padding"))
                padding = opIt->get<std::array<unsigned short, 4>>("padding");
            else
                padding = {0,0,0,0};

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
            auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
//            int newOutputSize = newOutputSizes.first;
//            int remainderOutputSize = newOutputSizes.second;

            unsigned startCoord = 0;
            for (std::size_t split = 0; split < numberOfSplits; split++)
            {
                tileShape tileStart({0,0,0,0,0});
                tileShape tileSize = inputShape;

                tileStart[axisToSplit] = startCoord;

                if (split == 0)
                    tileSize[axisToSplit] = inferInputSize(newOutputSizes[split],padStart,0,kernelSize,kernelStride);
                else if (split == (numberOfSplits-1))
                    tileSize[axisToSplit] = inferInputSize(newOutputSizes[split],0,padEnd,kernelSize,kernelStride);
                else
                    tileSize[axisToSplit] = inferInputSize(newOutputSizes[split],0,0,kernelSize,kernelStride);

                mv::Tiling newTile(tileStart, tileSize);
                setChildTile(newTile, split);

                // Compute start coordinates for the next tile
                // TODO: compute correct formula.
                if (split == 0)
                    startCoord += newOutputSizes[split] * kernelStride - (inferInputSize(newOutputSizes[split],0,0,kernelSize,kernelStride) - tileSize[axisToSplit]);
                else
                    startCoord += newOutputSizes[split] * kernelStride;
            }
        }

        void generateTiling(mv::Data::OpListIterator opIt)
        {
            if(axis_ == "K")
                generateWeightsTiling();
            else if (axis_ == "H")
                generateSpatialTiling(opIt);
            else if (axis_ == "W")
                generateSpatialTiling(opIt);
        }

        //TODO::build proper stream out of this
        void printOut(unsigned depth) const
        {
            std::cout << "Master : "; printShape(size_) ; std::cout << std::endl;

            for (auto& tile : childTiles_)
            {
                for (unsigned tab = 0; tab < depth; tab++)
                    std::cout<<"\t";

                std::cout << "\tChild: ";
                tile.printOut(depth+1);\
            }
        }
    };
}
#endif
