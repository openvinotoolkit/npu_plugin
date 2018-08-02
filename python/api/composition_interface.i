
/// SWIG Interface File.

// Include the external numpy swig bindings.
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%module composition_api
%{
    #include <include/mcm/compiler/compilation_unit.hpp>
    #include <include/mcm/computation/model/op_model.hpp>
    #include <math.h>

    int testSWIG(){
        /// A simple test to ensure the connection between Python and C++ is working
        int test = 1;
        return test;
    }

    mv::CompilationUnit* getCompilationUnit()
    {

        auto unit = new mv::CompilationUnit();
        unit->loadTargetDescriptor(mv::Target::ma2480);

        // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
        unit->compilationDescriptor()["GenerateDot"]["output"] = std::string("pycm.dot");
        unit->compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
        unit->compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
        unit->compilationDescriptor()["GenerateDot"]["html"] = true;
        unit->compilationDescriptor()["GenerateJSON"]["output"] = std::string("cpp.json");
        unit->compilationDescriptor()["GenerateBlob"]["output"] = std::string("cpp.blob");
        return unit;

    }

    mv::OpModel* getModel(mv::CompilationUnit *unit)
    {
        return new mv::OpModel(unit->model());
    }

    int compile(mv::CompilationUnit *unit)
    {
        unit->initialize();
        auto compOutput = unit->run();
        return (int)compOutput["passes"].last()["blobSize"].get<long long>();
    }

    mv::UnsignedVector2D * get2DVector(int x, int y){
        mv::UnsignedVector2D *a = new mv::UnsignedVector2D();
        a->e0 = x;
        a->e1 = y;
        return a;
    }


    mv::UnsignedVector4D * get4DVector(int w, int x, int y, int z){
        mv::UnsignedVector4D *a = new mv::UnsignedVector4D();
        a->e0 = w;
        a->e1 = x;
        a->e2 = y;
        a->e3 = z;
        return a;
    }

    mv::Shape * getShape(int x){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(x);
        return a;
    }

    mv::Shape * getShape(int x, int y){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(x, y);
        return a;
    }

    mv::Shape * getShape(int x, int y, int z){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(x, y, z);
        return a;
    }

    mv::Shape * getShape(int b, int x, int y, int z){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(b, x, y, z);
        return a;
    }

    mv::dynamic_vector<mv::float_type> * getData(float * d, size_t len){
        /// Populate a Vector with a numpy array.
        mv::dynamic_vector<mv::float_type> * weightsData = new mv::dynamic_vector<mv::float_type>(d, d + len);
        return weightsData;
    }

    int testConv(
        mv::Data::OpListIterator &target,
        unsigned exp_strideX,
        unsigned exp_strideY,
        unsigned exp_padX,
        unsigned exp_padY
        ){
        /// A couple of simple checks to ensure we have loaded the items correctly.

        int ret_val = 0;    // Success
        mv::UnsignedVector2D stride = target->getAttr("stride").getContent<mv::UnsignedVector2D>();
        mv::UnsignedVector4D pad = target->getAttr("padding").getContent<mv::UnsignedVector4D>();
        if(stride.e0 != exp_strideX)
            ret_val = 1;
        if(stride.e1 != exp_strideY)
            ret_val = 2;
        if(pad.e1 != exp_padX)
            ret_val = 3;
        if(pad.e3 != exp_padY)
            ret_val = 4;
        // TODO Consider assymetric padding

        return ret_val;
    }

    mv::Data::TensorIterator input(mv::OpModel *o, const mv::Shape &shape){
        /// Add an Input Layer to the OpModel and return the relevant iterator
        return o->input(shape, mv::DType::Float, mv::Order::Planar);
    }

    mv::Data::TensorIterator output(mv::OpModel *o, mv::Data::TensorIterator input){
        /// Add an Output Layer to the OpModel and return the relevant iterator
        return o->output(input);
    }

    mv::Data::TensorIterator maxpool2D(mv::OpModel *o, mv::Data::TensorIterator input, unsigned kernelSizeX,
        unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY){
        /// Add a Max Pooling Layer to the OpModel and return the relevant iterator
        return o->maxpool2D(input, {kernelSizeX, kernelSizeY}, {strideX, strideY},
            {padX, padX, padY, padY});
    }

    mv::Data::TensorIterator maxpool2D_caffe(mv::OpModel *o, mv::Data::TensorIterator input, unsigned kernelSizeX,
        unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY){

        /// This differs from the above because caffe calculates output sizes differently.
        /// To compensate, we add values to pad.
        /// See: https://github.com/BVLC/caffe/issues/1318

        int adj_X = 0, adj_Y = 0;

        mv::Shape i = input->getShape();

        float inner_x_calc = float(i[0] + padX + padX - kernelSizeX);
        float inner_y_calc = float(i[1] + padY + padY - kernelSizeY);

        float caffe_x = ceil(inner_x_calc / strideX) + 1;
        float caffe_y = ceil(inner_y_calc / strideX) + 1;

        float tensorflow_x = ceil((inner_x_calc +1) / strideX);
        float tensorflow_y = ceil((inner_y_calc +1) / strideX);

        adj_X = caffe_x - tensorflow_x;
        adj_Y = caffe_y - tensorflow_y;

        return o->maxpool2D(input, {kernelSizeX, kernelSizeY}, {strideX, strideY},
            {padX, padX+ adj_X, padY, padY + adj_Y});
    }

    mv::Data::TensorIterator concat(mv::OpModel *o, mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        /// Add a Concat Layer to the OpModel and return the relevant iterator.
        /// Allows only two inputs at a time. More must cascade
        return o->concat(input0, input1);
    }

    mv::Data::TensorIterator conv2D(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
        unsigned strideX, unsigned strideY, unsigned padX, unsigned padY){
        /// Add a Convolutional Layer to the OpModel and return the relevant iterator
        return o->conv2D(input, filters, {strideX, strideY}, {padX, padX, padY, padY});
    }

    mv::Data::TensorIterator conv2D_caffe(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
        unsigned strideX, unsigned strideY, unsigned padX, unsigned padY){
        /// This differs from the above because caffe calculates output sizes differently.
        /// To compensate, we add values to pad.

        int adj_X = 0, adj_Y = 0;

        mv::Shape i = input->getShape();
        mv::Shape k = filters->getShape();

        int kernelSizeX =  k[0];
        int kernelSizeY =  k[1];

        float inner_x_calc = float(i[0] + padX + padX - kernelSizeX);
        float inner_y_calc = float(i[1] + padY + padY - kernelSizeY);

        float caffe_x = ceil(inner_x_calc / strideX) + 1;
        float caffe_y = ceil(inner_y_calc / strideX) + 1;

        float tensorflow_x = ceil((inner_x_calc +1) / strideX);
        float tensorflow_y = ceil((inner_y_calc +1) / strideX);

        adj_X = caffe_x - tensorflow_x;
        adj_Y = caffe_y - tensorflow_y;

        return o->conv2D(input, filters, {strideX, strideY}, {padX , padX- adj_X, padY, padY - adj_Y});
    }

    mv::Data::TensorIterator constant(mv::OpModel *o, const mv::dynamic_vector<mv::float_type>& data, const mv::Shape &shape){
        /// Add a Constant Layer to the OpModel and return the relevant iterator
        return o->constant(data, shape, mv::DType::Float, mv::Order::Planar);
    }

    mv::Data::OpListIterator getSourceOp(mv::OpModel *o, mv::Data::TensorIterator tensor){
        // Get source operation of a tensor
        return o->getSourceOp(tensor);
    }

    mv::Data::TensorIterator matMul(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator weights){
        return o->matMul(input, weights);
    }

    mv::Data::TensorIterator avgpool2D(mv::OpModel *o, mv::Data::TensorIterator input, mv::UnsignedVector2D kernelSize, mv::UnsignedVector2D stride, mv::UnsignedVector4D padding){
        return o->avgpool2D(input, kernelSize, stride, padding);
    }

    mv::Data::TensorIterator avgpool2D_caffe(mv::OpModel *o, mv::Data::TensorIterator input, unsigned kernelSizeX,
        unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY){

        /// This differs from the above because caffe calculates output sizes differently.
        /// To compensate, we add values to pad.
        /// See: https://github.com/BVLC/caffe/issues/1318

        int adj_X = 0, adj_Y = 0;

        mv::Shape i = input->getShape();

        float inner_x_calc = float(i[0] + padX + padX - kernelSizeX);
        float inner_y_calc = float(i[1] + padY + padY - kernelSizeY);

        float caffe_x = ceil(inner_x_calc / strideX) + 1;
        float caffe_y = ceil(inner_y_calc / strideX) + 1;

        float tensorflow_x = ceil((inner_x_calc +1) / strideX);
        float tensorflow_y = ceil((inner_y_calc +1) / strideX);

        adj_X = caffe_x - tensorflow_x;
        adj_Y = caffe_y - tensorflow_y;

        return o->avgpool2D(input, {kernelSizeX, kernelSizeY}, {strideX, strideY},
            {padX, padX+ adj_X, padY, padY+ adj_Y});
    }

    mv::Data::TensorIterator batchNorm(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator mean, mv::Data::TensorIterator variance, mv::Data::TensorIterator offset, mv::Data::TensorIterator scale, mv::float_type varianceEps){
        return o->batchNorm(input, mean, variance, offset, scale, varianceEps);
    }
    mv::Data::TensorIterator scale(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator scale){
        return o->scale(input, scale);
    }
    mv::Data::TensorIterator relu(mv::OpModel *o,mv::Data::TensorIterator input){
        return o->relu(input);
    }
    mv::Data::TensorIterator softmax(mv::OpModel *o,mv::Data::TensorIterator input){
        return o->softmax(input);
    }
    mv::Data::TensorIterator add(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->add(input0, input1);
    }
    mv::Data::TensorIterator subtract(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->subtract(input0, input1);
    }
    mv::Data::TensorIterator multiply(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->multiply(input0, input1);
    }
    mv::Data::TensorIterator fullyConnected(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->fullyConnected(input0, input1);
    }

    mv::Data::TensorIterator divide(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->divide(input0, input1);
    }
    mv::Data::TensorIterator reshape(mv::OpModel *o,mv::Data::TensorIterator input, const mv::Shape& shape){
        return o->reshape(input, shape);
    }
    mv::Data::TensorIterator bias(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator bias_values){
        return o->bias(input, bias_values);
    }

    /*void produceDOT(mv::OpModel *o, const char *fileName){
        mv::TargetDescriptor dummyTargetDesc;
        mv::json::Object compDesc;
        mv::json::Object compOutput;
        std::string fileNameStr(fileName);
        compDesc["GenerateDot"]["output"] = fileNameStr + ".dot";
        compDesc["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
        compDesc["GenerateDot"]["content"] = std::string("full");
        compDesc["GenerateDot"]["html"] = true;
        mv::pass::PassRegistry::instance().find("GenerateDot")->run(*o, dummyTargetDesc, compDesc, compOutput);
        std::string cmd = "dot -Tsvg " + fileNameStr + ".dot -o " + fileNameStr + ".svg";
        system(cmd.c_str());
    }*/

 %}

#include <include/mcm/computation/model/op_model.hpp>
#include <include/mcm/computation/model/control_model.hpp>
#include <include/mcm/deployer/serializer.hpp>

// The section below is exposing the functions within the included files,
// or the ones defined above in the module.

namespace mv
{

    class OpModel
    {
    public:
        bool isValid() const;
    };

    namespace Data
    {
        class TensorIterator
        {
        public:
            ~TensorIterator();
        };

        class OpListIterator
        {
        public:
            ~OpListIterator();
        };

    }
}

int testSWIG();
mv::CompilationUnit* getCompilationUnit();
mv::OpModel* getModel(mv::CompilationUnit *unit);
int compile(mv::CompilationUnit *unit);
mv::Shape * getShape(int x);
mv::Shape * getShape(int x, int y);
mv::Shape * getShape(int x, int y, int z);
mv::Shape * getShape(int b, int x, int y, int z);
mv::UnsignedVector2D * get2DVector(int x, int y);
mv::UnsignedVector4D * get4DVector(int w, int x, int y, int z);

// Expand a numpy array to a data pointer and a length
%include "stdint.i"
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* d, int len)}
mv::dynamic_vector<mv::float_type> * getData(float * d, int len);


mv::Data::TensorIterator input(mv::OpModel * o, const mv::Shape &shape);
mv::Data::TensorIterator output(mv::OpModel * o, mv::Data::TensorIterator input);
mv::Data::TensorIterator conv2D(mv::OpModel * o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
    unsigned strideX, unsigned strideY, unsigned padX, unsigned padY);
mv::Data::TensorIterator conv2D_caffe(mv::OpModel * o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
    unsigned strideX, unsigned strideY, unsigned padX, unsigned padY);
mv::Data::TensorIterator maxpool2D(mv::OpModel * o, mv::Data::TensorIterator input, unsigned kernelSizeX,
    unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY);
mv::Data::TensorIterator maxpool2D_caffe(mv::OpModel * o, mv::Data::TensorIterator input, unsigned kernelSizeX,
    unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY);
mv::Data::TensorIterator avgpool2D_caffe(mv::OpModel *o, mv::Data::TensorIterator input, unsigned kernelSizeX,
        unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY);
mv::Data::TensorIterator concat(mv::OpModel * o, mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::OpListIterator getSourceOp(mv::OpModel *o, mv::Data::TensorIterator tensor);

mv::Data::TensorIterator matMul(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator weights);
mv::Data::TensorIterator avgpool2D(mv::OpModel *o, mv::Data::TensorIterator input, mv::UnsignedVector2D kernelSize, mv::UnsignedVector2D stride, mv::UnsignedVector4D padding);
mv::Data::TensorIterator batchNorm(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator mean, mv::Data::TensorIterator variance, mv::Data::TensorIterator offset, mv::Data::TensorIterator scale, float varianceEps);
mv::Data::TensorIterator scale(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator scale);
mv::Data::TensorIterator relu(mv::OpModel *o,mv::Data::TensorIterator input);
mv::Data::TensorIterator softmax(mv::OpModel *o,mv::Data::TensorIterator input);
mv::Data::TensorIterator add(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator subtract(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator multiply(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator divide(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator reshape(mv::OpModel *o,mv::Data::TensorIterator input, const mv::Shape& shape);
mv::Data::TensorIterator bias(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator bias_values);
mv::Data::TensorIterator fullyConnected(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator constant(mv::OpModel * o, const mv::dynamic_vector<mv::float_type>& data, const mv::Shape &shape);

int testConv(
    mv::Data::OpListIterator &target,
    int exp_strideX,
    int exp_strideY,
    int exp_padX,
    int exp_padY
);
