#ifndef TILING_HPP_
#define TILING_HPP_

#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{
    class Tiling {
    private:
        using TileShape = std::vector<std::size_t>;

        enum TileDim { TILE_DIM_W, TILE_DIM_H, TILE_DIM_C, TILE_DIM_K, TILE_DIM_N };
        TileShape start_;
        TileShape size_;

        std::string axis_;
        std::vector<Tiling> childTiles_;

        void printShape(const TileShape& shape) const
        {
            std::cout<< "{";
            for(size_t i = 0; i <= TILE_DIM_N; ++i)
                std::cout<< shape[i] << "," ;
            std::cout<<"}";
        }

    public:

        Tiling() : start_({0,0,0,0,0}), size_({0,0,0,0,0}), axis_(""), childTiles_(0) {}
        Tiling(const Shape& actShape, const Shape& kernelShape)
                : start_({0,0,0,0,0}), axis_(""), childTiles_(0)
        {
            size_.resize(5);
            size_[TILE_DIM_W] = actShape[mv::IO_WIDTH_DIMENSION];
            size_[TILE_DIM_H] = actShape[mv::IO_HEIGHT_DIMENSION];
            size_[TILE_DIM_C] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_K] = kernelShape[mv::KERNEL_OUTPUT_CHANNELS];
            size_[TILE_DIM_N] = actShape[mv::IO_BATCH_DIMENSION];
        }
        Tiling(const Shape& actShape)
                : start_({0,0,0,0,0}),axis_(""), childTiles_(0)
        {
            size_.resize(5);
            size_[TILE_DIM_W] = actShape[mv::IO_WIDTH_DIMENSION];
            size_[TILE_DIM_H] = actShape[mv::IO_HEIGHT_DIMENSION];
            size_[TILE_DIM_C] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_K] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_N] = actShape[mv::IO_BATCH_DIMENSION];
        }

        Tiling(const TileShape& start, const TileShape& size) :
                start_(start),size_(size),axis_(""),childTiles_(0) {}

        Tiling(const std::string& axis, std::size_t tiles)
                : start_({0,0,0,0,0}), size_({0,0,0,0,0}), axis_(axis), childTiles_(tiles)
        {
        }

        Tiling(const Shape& start, Shape& size, std::string axis, std::size_t childTiles)
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

        const std::string& getAxis() const { return axis_; }
        void setAxis(const std::string& axis) { axis_ = axis; }

        const TileShape& getStartCoord() const { return start_; }
        void setStartCoord(const TileShape& start) { start_ = start; }

        const TileShape& getSize() const { return size_; }
        void setSize(const TileShape& size) { size_ = size; }

        // TODO: This method requires const correctness, but it can't be changed because 
        // spatial_split_streaming has already started using it. Will be replaced by
        // getChildTiles().
        std::vector<Tiling>& childTiles() { return childTiles_; }
        const std::vector<Tiling>& getChildTiles() const { return childTiles_; }
        void setChildTile(const Tiling& tile, unsigned index) { childTiles_[index] = tile; }

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

        static inline int inferInputSize(int outputSize, int padding_start, int padding_end, int kernel_size, int kernel_stride)
        {
            int inputSize =  ((outputSize -1) * kernel_stride)  -padding_start - padding_end + kernel_size;
            return inputSize;
        }

        static inline int inferOutputSize(int inputSize, int padding_start, int padding_end, int kernel_size, int kernel_stride)
        {
            int outputSize = ( inputSize + padding_start + padding_end - kernel_size) / kernel_stride + 1;
            return outputSize;
        }

        void generateWeightsTiling()
        {
            auto numberOfSplits = childTiles_.size();
            auto parentTileShape = getSize();
            auto axisToSplit = mv::Shape::getAxis(getAxis());

            int newSize = ceil(((double)parentTileShape[axisToSplit]) / ((double)numberOfSplits));
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
                TileShape tileStart({0,0,0,0,0});
                TileShape tileSize = parentTileShape;

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

            int padStart=0,padEnd=0;

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

            unsigned startCoord = 0;
            for (std::size_t split = 0; split < numberOfSplits; split++)
            {
                TileShape tileStart({0,0,0,0,0});
                TileShape tileSize = inputShape;

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

        void generateBatchTiling()
        {
            auto numberOfSplits = childTiles().size();
            auto inputShape = getSize();
            // MV::Shape considers N(batches) at the same index as K(out channels for weight sets)
            // which won't work with our tiling logic where N and K sizes jave separate entries
            auto axisToSplit =  TILE_DIM_N;

            auto newInputSizes = tileSpatialOutputSize(inputShape[axisToSplit], numberOfSplits);

            unsigned startCoord = 0;
            for (std::size_t split = 0; split < numberOfSplits; split++)
            {
                TileShape tileStart({0,0,0,0,0});
                TileShape tileSize = inputShape;

                tileStart[axisToSplit] = startCoord;
                tileSize[axisToSplit] = newInputSizes[split];

                mv::Tiling newTile(tileStart, tileSize);
                setChildTile(newTile, split);

                // Compute start coordinates for the next tile
                startCoord += newInputSizes[split];
            }
        }

        void generateTiling(mv::Data::OpListIterator opIt)
        {
            if(axis_ == "K" || axis_ == "C")
                generateWeightsTiling();
            else if (axis_ == "H" || axis_ == "W")
                generateSpatialTiling(opIt);
            else if (axis_ == "N")
                generateBatchTiling();
        }

        void print(std::ostream& o, const Tiling& tiling, int depth = 0) const {
            o << std::string(depth, '\t') << "{";
            for (std::size_t i = 0; i < tiling.getSize().size(); ++i)
            {
                if (i != 0) o << ",";
                o << tiling.getSize()[i];
            }
            o << "}" << std::endl;
            for (const auto& childTile : tiling.getChildTiles())
            {
                print(o, childTile, depth + 1);
            }
        }
    };
    // TODO: This currently leads to compile errors due to way in which the tiling.hpp
    // gets included. Uncomment this once this is resolved.
    /*
    std::ostream& operator<<(std::ostream& o, const Tiling& t) {
        t.print(o, t);
        return o;
    }
    */
}
#endif
