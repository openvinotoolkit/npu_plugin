#ifndef INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_
#define INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_

#include "include/mcm/api/compositional_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/stl_allocator.hpp"
#include "include/mcm/computation/model/iterator/group_context.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

namespace mv
{

class CompositionalModelRecorder : public ComputationModel, public CompositionalModel
{

	OpModel& modelRef_;
	std::ofstream outputSourceFile; /*Recorded source file*/
	std::ofstream outputWeightsFile; /*Recorded weights file*/
	std::ostringstream ss; /*Stream for source file*/
	std::ostringstream ws; /*Stream for weights file*/
	string recordedSourceFileName;
	string recordedWeghtsFileName;
	string recordedSourceFileNameCpp;
	static const std::string savedRecordingsPath_;

	int inputCounter = 0;
	int weightsVectorCounter = 0;
	int constantCounter = 0;
	int batchNormCounter = 0;
	int reluCounter = 0;
	int maxpool2DCounter = 0;
	int outputCounter = 0;
	int addCounter = 0;
	int fullyConnectedCounter = 0;
	int softmaxCounter = 0;
	int multiplyCounter = 0;
	int reshapeCounter = 0;
	int avgpool2DCounter = 0;
	int conv2DCounter = 0;
	int concatCounter = 0;
	int subtractCounter = 0;
	int divideCounter = 0;
	int biasCounter = 0;
	int scaleCounter = 0;

public:
	CompositionalModelRecorder(Logger::VerboseLevel verboseLevel, bool logTime, OpModel& model);
	~CompositionalModelRecorder();
	Data::TensorIterator input(const Shape& shape, DType dType, Order order, const string& name = "") override;
	Data::TensorIterator output(Data::TensorIterator input, const string& name = "") override;
	Data::TensorIterator constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name = "") override;
	Data::TensorIterator conv2D(Data::TensorIterator input, Data::TensorIterator filters, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") override;
	Data::TensorIterator matMul(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator maxpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") override;
	Data::TensorIterator avgpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") override;
	Data::TensorIterator concat(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator batchNorm(Data::TensorIterator input, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, float_type varianceEps, const string& name = "") override;
	Data::TensorIterator scale(Data::TensorIterator input, Data::TensorIterator scale, const string& name = "") override;
	Data::TensorIterator relu(Data::TensorIterator input, const string& name = "") override;
	Data::TensorIterator softmax(Data::TensorIterator input, const string& name = "") override;
	Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, const string& name = "") override;
	Data::TensorIterator bias(Data::TensorIterator input, Data::TensorIterator biases, const string& name = "") override;
	Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, const string& name = "") override;
	void createAndOpenFile();
	void closeFile();
	void writeWeightsToFile(const dynamic_vector<float_type>& weightsData, string weightsVectorName);

	bool isValid() const override;
	bool isValid(const Data::TensorIterator& it) const override;
	bool isValid(const Data::OpListIterator& it) const override;

	Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;
	bool addAttr(Data::OpListIterator op, const string& name, const Attribute& attr) override;


};

/*Non-member functions to improve encapsulation*/
string dtypeToString(DType dType);
string orderToString(Order order);


/*Convert Vector2D to a string*/
template <class T>
string vector2DToString(T const& vec)
{
    std::ostringstream s;

    s << "{" << vec.e0 << "," << vec.e1 << "}";

    return s.str();
}

/*Convert Vector2D to a string*/
template <class T>
string vector4DToString(T const& vec)
{
    std::ostringstream s;

    s << "{" << vec.e0 << "," << vec.e1 << "," << vec.e2 << "," << vec.e3 << "}";

    return s.str();
}

}
#endif /* INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_ */
