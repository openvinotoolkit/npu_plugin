
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
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::TextFormat;

/*
layer {
  name: "conv1"
  type: "Convolution" ??
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
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
    mv::CompositionalModel& cm = unit.model();

    const char *filename = "./test.prototxt";
    caffe::NetParameter param;
    caffe::LayerParameter lparam;
    caffe::ConvolutionParameter cparam;
    std::string data;

    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
    cout << "File not found: " << filename;


    auto input = cm.input({224, 224, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajorPlanar);
    mv::Shape kernelShape = {7, 7, 3, 64};
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = cm.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::array<unsigned short, 2> stride =  {2, 2};
    std::array<unsigned short, 4> padding = {3, 3, 3, 3};
    auto conv = cm.conv2D(input, weights, stride, padding);
    cm.output(conv);

    //TODO: cast
    mv::OpModel opModel(cm);

    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == mv::OpType::Conv2D)
        {
            lparam.set_name(opIt->getName());
            
            /*get parent op*/
            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));
            lparam.add_bottom(parentOpIt->getName());

            /*get sink op*/
            auto sourceOpIt = opIt.leftmostChild();
            lparam.add_top(sourceOpIt->getName());

            /*get stride*/
            cparam.add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);

            /*get kernel*/
             //cparam.add_stride(opIt->get<std::array<unsigned short, 2>>("kernel")[0]);

             std::vector<std::string> attrKeys(opIt->attrsKeys());
            for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
            std::cout << opIt->get(*attrIt).toString() << std::endl; 

        }
    } 
    
    

    /*create prototxt*/
    std::ofstream ofs; 
    ofs.open ("test.prototxt", std::ofstream::out | std::ofstream::app);
    ofs << lparam.Utf8DebugString(); 
    ofs << cparam.Utf8DebugString(); 
    ofs.close();




            /* std::vector<std::string> attrKeys(opIt->attrsKeys());
            for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
            std::cout << opIt->get(*attrIt).toString() << std::endl;  */

    //google::protobuf::io::FileOutputStream* output = new google::protobuf::io::FileOutputStream(fd);


     /* lparam.set_name("AlexNet");
     param.add_input_dim(0);
     param.add_input_dim(1);
     param.add_input_dim(2);
     param.set_input_dim(0, 3);
     param.set_input_dim(1, 3); */
    
//     param.set_input_dim(2, 3);

//     //param.SerializeToString(&data);

//     //std::cout << param.DebugString();

//     //bool success = google::protobuf::TextFormat::Print(data, output);

//     //std::cout << "bool is " << success << std::endl;

//     /* bool success = google::protobuf::TextFormat::PrintToString(param, &data);
//     std::cout << "bool is " << success << std::endl;
//     printf("%s", data.c_str());


//     bool success1 = google::protobuf::TextFormat::Print(param, output);
//     std::cout << "bool is " << success1 << std::endl; */
// //     cout << "Input size: " << param.input_size() << endl;
// //     cout << "Input shape size: " << param.input_shape_size() << endl;
// //     cout << "Input dimension size: " << param.input_dim_size() << endl;
// //     cout << "Input dim: " << param.input_dim(1) << endl;

// //      for (int j = 0; j < param.input_dim_size(); j++) {
// //     cout << "Input Dim "<< j << ": " << param.input_dim(j) << endl;
// //   } 

// //param.SerializeToFileDescriptor(fd);
//     //CHECK(google::protobuf::TextFormat::Print(data, output));





//   output->Flush();
//     close(fd);
    return 0;
}


