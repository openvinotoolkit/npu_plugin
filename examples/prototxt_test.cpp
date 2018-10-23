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


int main()
{

    
    caffe::NetParameter param;

    std::string data;

    const char *filename = "./test.prototxt";

    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
        cout << "File not found: " << filename;

    google::protobuf::io::FileOutputStream* output = new google::protobuf::io::FileOutputStream(fd);

    param.set_name("AlexNet");
    param.add_input_dim(0);
    param.add_input_dim(1);
    param.add_input_dim(2);
    param.set_input_dim(0, 3);
    param.set_input_dim(1, 3);
    
    param.set_input_dim(2, 3);

    //param.SerializeToString(&data);

    //std::cout << param.DebugString();

    //bool success = google::protobuf::TextFormat::Print(data, output);

    //std::cout << "bool is " << success << std::endl;

    bool success = google::protobuf::TextFormat::PrintToString(param, &data);
    std::cout << "bool is " << success << std::endl;
    printf("%s", data.c_str());


    bool success1 = google::protobuf::TextFormat::Print(param, output);
    std::cout << "bool is " << success << std::endl;
//     cout << "Input size: " << param.input_size() << endl;
//     cout << "Input shape size: " << param.input_shape_size() << endl;
//     cout << "Input dimension size: " << param.input_dim_size() << endl;
//     cout << "Input dim: " << param.input_dim(1) << endl;

//      for (int j = 0; j < param.input_dim_size(); j++) {
//     cout << "Input Dim "<< j << ": " << param.input_dim(j) << endl;
//   } 

//param.SerializeToFileDescriptor(fd);
    //CHECK(google::protobuf::TextFormat::Print(data, output));




 
    close(fd);
    return 0;
}

   
