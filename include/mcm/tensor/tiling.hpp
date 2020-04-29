#ifndef TILING_HPP_
#define TILING_HPP_

#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include <vector>
#include "include/mcm/computation/model/data_model.hpp"


namespace mv
{
    class Tiling {
    private:
        Shape start_;
        Shape size_;

        std::string axis_;
        std::vector<Tiling> childTiles_;

    public:

        Tiling() :start_({0,0,0,0}), size_({0,0,0,0}), axis_(""), childTiles_(0) {}
        Tiling( Shape& start, Shape& size)
                : start_(start), size_(size), axis_(""), childTiles_(0)
        {
        }

        Tiling( std::string& axis, std::size_t tiles)
                : start_({0,0,0,0}), size_({0,0,0,0}), axis_(axis), childTiles_(tiles)
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

        Shape& getStartCoord() { return start_; }
        void setStartCoord(Shape start) { start_ = start; }

        Shape& getEndCoord() { return start_; }
        void setEndCoord(Shape start) { start_ = start; }

        Shape& getSize() { return size_; }
        void setSize(Shape size) { size_ = size; }

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

        void generateWeightsTiling()
        {
            auto numberOfSplits = childTiles_.size();
            auto parentTileShape = getSize();
            auto axisToSplit =  mv::Shape::getAxis(getAxis());
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
                mv::Tiling newTile(tileStart,tileSize);
                setChildTile(newTile,split);
            }
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
            else if (axisToSplit == mv::Shape::getAxis("H") || axisToSplit == mv::Shape::getAxis("N"))
            {
                padStart = padding[2];
                padEnd = padding[3];
            }

            int outputSize =  inferOutputSize(inputShape[axisToSplit],padStart,padEnd,kernelSize,kernelStride);
            std::vector<size_t> sizes = tileSpatialOutputSize(outputSize, numberOfSplits);

            unsigned startCoord = 0;
            for (std::size_t split = 0; split < numberOfSplits; split++)
            {
                mv::Shape tileStart({0,0,0,0});
                mv::Shape tileSize = inputShape;

                tileStart[axisToSplit] = startCoord;

                if (split == 0)
                    tileSize[axisToSplit] = inferInputSize(sizes[split],padStart,0,kernelSize,kernelStride);
                else if (split == (numberOfSplits-1))
                    tileSize[axisToSplit] = inferInputSize(sizes[split],0,padEnd,kernelSize,kernelStride);
                else
                    tileSize[axisToSplit] = inferInputSize(sizes[split],0,0,kernelSize,kernelStride);

                mv::Tiling newTile(tileStart, tileSize);
                setChildTile(newTile, split);

                // Compute start coordinates for the next tile
                // TODO: compute correct formula.
                if (split == 0)
                    startCoord += sizes[split] * kernelStride - (inferInputSize(sizes[split],0,0,kernelSize,kernelStride) - tileSize[axisToSplit]);
                else
                    startCoord += sizes[split] * kernelStride;
            }
        }
    };
}
#endif
