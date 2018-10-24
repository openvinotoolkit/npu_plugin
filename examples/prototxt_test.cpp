
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

    /*Network object*/
    caffe::NetParameter netParam;

    /*Create computation model*/
    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajorPlanar);
    mv::Shape kernelShape = {3, 3, 3, 3};
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = cm.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::array<unsigned short, 2> stride = {2, 2};
    std::array<unsigned short, 4> padding = {3, 3, 3, 3};
    auto conv = cm.conv2D(input, weights, stride, padding);
    auto softmax = cm.softmax(conv);
    cm.output(softmax);

    mv::OpModel &opModel = dynamic_cast<mv::OpModel &>(cm);

    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == mv::OpType::Input)
        {
            /*Don't add layer for input*/

            caffe::InputParameter inputParam;

            /*Set name and type of the layer*/
            //inputParam.set_name("Input_0");

             /* add input dimensions*/
            //caffe::BlobShape *blobShape = inputParam.add_input_shape();
            netParam.add_input("Input_0");
            netParam.add_input_dim(0);
            netParam.add_input_dim(1);
            netParam.add_input_dim(2);
            netParam.add_input_dim(3);

            netParam.set_input_dim(0, 1);
            netParam.set_input_dim(1, 3);
            netParam.set_input_dim(2, 224);
            netParam.set_input_dim(3, 224);

            //TODO: Move this to function 
            /*create caffemodel*/
            fstream output("weights.caffemodel", ios::out | ios::binary);
            netParam.SerializeToOstream(&output);

            /*remove blob before writing to prototxt*/
            //layerParam->clear_blobs();

            /*create prototxt*/
            std::ofstream ofs;
            ofs.open("test.prototxt", std::ofstream::out | std::ofstream::trunc);
            ofs << netParam.Utf8DebugString();
            ofs.close();
            

        }
    
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

            /* add weights*/
            caffe::BlobProto *blobProto = layerParam->add_blobs();
            caffe::BlobShape *blobShape = blobProto->mutable_shape();

            blobShape->add_dim(0);
            blobShape->add_dim(1);
            blobShape->add_dim(2);
            blobShape->add_dim(3);

            blobShape->set_dim(0, parentOpIt1->get<mv::Shape>("shape")[3]);
            blobShape->set_dim(1, parentOpIt1->get<mv::Shape>("shape")[2]);
            blobShape->set_dim(2, parentOpIt1->get<mv::Shape>("shape")[1]);
            blobShape->set_dim(3, parentOpIt1->get<mv::Shape>("shape")[0]);

            blobProto->clear_double_data();
            blobProto->clear_double_diff();

            /*ColumnMajor is format for caffemodel*/
            weights->setOrder(mv::OrderType::ColumnMajor);
            std::vector<double> caffeModelWeights = (*weights).getData();

            for (unsigned i = 0; i < caffeModelWeights.size(); ++i)
            {
                blobProto->add_double_data(weightsData[i]);

            //TODO: Move this to function 
            /*create caffemodel*/
            fstream output("weights.caffemodel", ios::out | ios::binary);
            netParam.SerializeToOstream(&output);

            /*remove blob before writing to prototxt*/
            layerParam->clear_blobs();

            /*create prototxt*/
            std::ofstream ofs;
            ofs.open("test.prototxt", std::ofstream::out | std::ofstream::trunc);
            ofs << netParam.Utf8DebugString();
            ofs.close();  // Note: May need to use this here (need to verify if weights are in correct order with mcmCheck):
                // void set_double_data(int index, double value);
            }


            //TODO: Move this to function 
            /*create caffemodel*/
            fstream output("weights.caffemodel", ios::out | ios::binary);
            netParam.SerializeToOstream(&output);

            /*remove blob before writing to prototxt*/
            layerParam->clear_blobs();

            /*create prototxt*/
            std::ofstream ofs;
            ofs.open("test.prototxt", std::ofstream::out | std::ofstream::trunc);
            ofs << netParam.Utf8DebugString();
            ofs.close();
        }

        if (opIt->getOpType() == mv::OpType::Softmax)
        {
            caffe::LayerParameter *layerParam = netParam.add_layer();

            /*Set name and type of the layer*/
            layerParam->set_name(opIt->getName());
            layerParam->set_type("Softmax");

            /*Get the input operation*/
            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));
            layerParam->add_bottom(parentOpIt->getName());

            /*Get the output operation*/
            auto sourceOpIt = opIt.leftmostChild();
            layerParam->add_top(sourceOpIt->getName());

            /*Set layer to have a softmax parameter*/
            //caffe::SoftmaxParameter *softmaxParam = layerParam->mutable_softmax_param();

            //TODO: Move this to function 
            /*create caffemodel*/
            fstream output("weights.caffemodel", ios::out | ios::binary);
            netParam.SerializeToOstream(&output);

            /*remove blob before writing to prototxt*/
            layerParam->clear_blobs();

            /*create prototxt*/
            std::ofstream ofs;
            ofs.open("test.prototxt", std::ofstream::out | std::ofstream::trunc);
            ofs << netParam.Utf8DebugString();
            ofs.close();
        }
    }

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("prototxt.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("prototext.blob");
    unit.compilationDescriptor()["GenerateProto"]["output"] = std::string("prototxt.txt");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    // Initialize compilation
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);

    // Run all passes
    unit.run();

    system("dot -Tsvg prototxt.dot -o cm_protoxt.svg");
    //system("dot -Tsvg cm_resnet18_adapt.dot -o cm_resnet18_adapt.svg");
    //system("dot -Tsvg cm_resnet18_final.dot -o cm_resnet18_final.svg");

    return 0;
}
