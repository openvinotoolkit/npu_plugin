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
    #include <string>


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
        mv::vector<mv::float_type> * weightsData = new mv::vector<mv::float_type>(d, len);
        return weightsData;
    }

    mv::ConstantTensor * getConstantTensor(mv::Shape * s, mv::vector<mv::float_type> data ){
        mv::ConstantTensor * a = new mv::ConstantTensor(*s, mv::DType::Float, mv::Order::NWHC, data);
        return a;
    }

    int testConv(
        mv::DataContext::OpListIterator &target,
        int exp_strideX,
        int exp_strideY,
        int exp_padX,
        int exp_padY
        ){

        int ret_val = 0;    // Success
        if(target->getAttr("strideX").getContent<mv::byte_type>() != exp_strideX)
            ret_val = 1;
        if(target->getAttr("strideY").getContent<mv::byte_type>() != exp_strideY)
            ret_val = 2;
        if(target->getAttr("padX").getContent<mv::byte_type>() != exp_padX)
            ret_val = 3;
        if(target->getAttr("padY").getContent<mv::byte_type>() != exp_padY)
            ret_val = 4;

        return ret_val;
    }

    mv::DataContext::OpListIterator input(mv::OpModel * o, const mv::Shape &shape){
        return o->input(shape, mv::DType::Float, mv::Order::NWHC);
    }

    mv::DataContext::OpListIterator output(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor){
        return o->output(predecessor);
    }

    mv::DataContext::OpListIterator maxpool(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::Shape &shape,
        uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY){

        return o->maxpool(predecessor, shape, strideX, strideY, padX, padY);
    }

    mv::DataContext::OpListIterator concat(mv::OpModel * o, mv::DataContext::OpListIterator &in0, mv::DataContext::OpListIterator &in1){
        return o->concat(in0, in1);
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
    public:
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
mv::DataContext::OpListIterator maxpool(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::Shape &shape,
    uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY);
mv::DataContext::OpListIterator concat(mv::OpModel * o, mv::DataContext::OpListIterator &in0, mv::DataContext::OpListIterator &in1);


int testConv(
    mv::DataContext::OpListIterator &target,
    int exp_strideX,
    int exp_strideY,
    int exp_padX,
    int exp_padY
);