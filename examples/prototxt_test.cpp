
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>
#include <algorithm>
#include <fstream> // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <string>
#include "caffe.pb.h"
#include <iostream>
#include <caffe/caffe.hpp>

using namespace std;
using google::protobuf::Message;
using google::protobuf::TextFormat;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

/*
layer {
  name: "conv1"
  type: "Convolution" ??
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
  }
}
*/

int main()
{
    mv::Logger::setVerboseLevel(mv::Logger::VerboseLevel::VerboseDebug);

    // Define the primary compilation unit
    mv::CompilationUnit unit("Test");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel &cm = unit.model();

    caffe::NetParameter netParam; /*Network object*/
   
    /*Create computation model*/
    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajorPlanar);
    mv::Shape kernelShape = {7, 7, 3, 64};
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = cm.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::array<unsigned short, 2> stride = {2, 2};
    std::array<unsigned short, 4> padding = {3, 3, 3, 3};
    auto conv = cm.conv2D(input, weights, stride, padding);
    cm.output(conv);

    mv::OpModel &opModel = dynamic_cast<mv::OpModel &>(cm);

    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == mv::OpType::Conv2D)
        {

            caffe::LayerParameter *layerParam = netParam.add_layer();

            /*Set name and type of the layer*/
            layerParam->set_name(opIt->getName());
            layerParam->set_type("Convolution");

            /*Get the input operation*/
            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));
            layerParam->add_bottom(parentOpIt->getName());

            /*Get the output operation*/
            auto sourceOpIt = opIt.leftmostChild();
            layerParam->add_top(sourceOpIt->getName());

            /*Set layer to have a conv parameter*/
            caffe::ConvolutionParameter *convParam = layerParam->mutable_convolution_param();
            
            /*Set stride on ConvolutionParameter object*/
            convParam->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);

            /*Set kernel on ConvolutionParameter object*/
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));
            convParam->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);

            /*Set number of output channels*/
            convParam->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);

            /*create prototxt*/
            std::ofstream ofs;
            ofs.open("test.prototxt", std::ofstream::out | std::ofstream::app);
            ofs << netParam.Utf8DebugString();
            ofs.close();
        }
    }

    /*create caffemodel*/
    //stream output("weights.caffemodel", ios::out | ios::trunc | ios::binary);
    //SerializeToOstream(&output));

    return 0;
}
