%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

 %module composition_api
 %{
    #include <include/fathom/computation/model/op_model.hpp>
    #include <include/fathom/computation/model/iterator/model_iterator.hpp>
    #include <include/fathom/computation/tensor/shape.hpp>
    #include <include/fathom/computation/tensor/constant.hpp>
    #include <include/fathom/computation/model/attribute.hpp>

    // mv::OpModel om;

    int testSWIG(){
        int test = 1;
        return test;
    }

    mv::OpModel * getOM(){
        mv::OpModel *om = new mv::OpModel();
        return om;
    }

    mv::Shape * getShape(int w, int x, int y, int z){
        mv::Shape* a = new mv::Shape(w, x, y, z);
        return a;
    }

    mv::vector<mv::float_type> * getData(float * d, size_t len){
        mv::vector<mv::float_type> * weightsData = new mv::vector<mv::float_type>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
            15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f
        };
        return weightsData;
    }

    mv::ConstantTensor * getConstantTensor(mv::Shape * s, mv::vector<mv::float_type> data ){
        mv::ConstantTensor * a = new mv::ConstantTensor(*s, mv::DType::Float, mv::Order::NWHC, data);
        return a;
    }

    int getAttrByte(mv::Attribute a){
        return a.getContent<mv::byte_type>();
    }

    mv::DataContext::OpListIterator input(mv::OpModel * o, const mv::Shape &shape){
        return o->input(shape, mv::DType::Float, mv::Order::NWHC);
    }

    mv::DataContext::OpListIterator output(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor){
        return o->output(predecessor);
    }

    mv::DataContext::OpListIterator conv(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::ConstantTensor &weights,
        uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY){

        return o->conv(predecessor, weights, strideX, strideY, padX, padY);
    }

 %}

#include <include/fathom/computation/model/op_model.hpp>
#include <include/fathom/computation/model/iterator/model_iterator.hpp>
#include <include/fathom/computation/model/iterator/data_context.hpp>
#include <include/fathom/computation/tensor/shape.hpp>
#include <include/fathom/computation/tensor/constant.hpp>
#include <include/fathom/computation/model/attribute.hpp>
#include <include/fathom/computation/api/compositional_model.hpp>
#include <include/fathom/computation/model/model.hpp>
#include <include/fathom/computation/op/input.hpp>
#include <include/fathom/computation/op/output.hpp>
#include <include/fathom/computation/op/conv.hpp>
#include <include/fathom/computation/op/maxpool.hpp>
#include <include/fathom/computation/op/concat.hpp>
#include <string>

namespace mv
{
    class OpModel : public ComputationModel, public CompositionalModel
    {

        // bool updateControlFlow_(DataContext::OpListIterator &newOp);

    public:

        // OpModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        // OpModel(Logger &logger);
        // OpModel(const ComputationModel &model);

        // DataContext::OpListIterator switchContext(ControlContext::OpListIterator &other);

        // DataContext::OpListIterator getInput();
        // DataContext::OpListIterator getOutput();
        // DataContext::OpListIterator opEnd();

        // mv::DataContext::OpListIterator input(const mv::Shape &shape, mv::DType dType, mv::Order order, const std::string &name = "");
        // DataContext::OpListIterator output(DataContext::OpListIterator &predecessor, const string &name = "");
        // DataContext::OpListIterator conv(DataContext::OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        // DataContext::OpListIterator maxpool(DataContext::OpListIterator &predecessor, const Shape &kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name = "");
        // DataContext::OpListIterator concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name = "");
        // bool addAttr(DataContext::OpListIterator &op, const string &name, const Attribute &attr);
        bool isValid() const;

    };
}


int testSWIG();
mv::OpModel * getOM();
mv::Shape * getShape(int w, int x, int y, int z);
mv::ConstantTensor * getConstantTensor(mv::Shape * s, mv::vector<mv::float_type> data );

%include "stdint.i"

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* d, int len)}
mv::vector<mv::float_type> * getData(float * d, int len);


mv::DataContext::OpListIterator input(mv::OpModel * o, const mv::Shape &shape);
mv::DataContext::OpListIterator output(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor);
mv::DataContext::OpListIterator conv(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::ConstantTensor &weights,
    uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY);