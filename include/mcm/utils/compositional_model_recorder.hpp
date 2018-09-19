#ifndef INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_
#define INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_

#include "include/mcm/api/compositional_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/stl_allocator.hpp"
#include "include/mcm/computation/model/iterator/group_context.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

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
	const string savedRecordingsPath_;
	unsigned weightsVectorCounter = 0;

public:
	CompositionalModelRecorder(OpModel& model, string recordingsPath_);
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
	Data::TensorIterator prelu(Data::TensorIterator input, Data::TensorIterator negative_slope, const string& name = "") override;
	Data::TensorIterator softmax(Data::TensorIterator input, const string& name = "") override;
	Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") override;
	Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, const string& name = "") override;
	Data::TensorIterator bias(Data::TensorIterator input, Data::TensorIterator biases, const string& name = "") override;
	Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, const string& name = "") override;
	void createRecordedSourceFiles(); /*Create two files to store the generated source code and weight vector definitions*/
	void completeRecordedSourceFile(); /*Populate the 'compilation passes' and end of the source file */
	void writeWeightsToFile(const dynamic_vector<float_type>& weightsData, string weightsVectorName);

	bool isValid() const override;
	bool isValid(const Data::TensorIterator& it) const override;
	bool isValid(const Data::OpListIterator& it) const override;

	Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;
	bool addAttr(Data::OpListIterator op, const string& name, const Attribute& attr) override;


};
}
#endif /* INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_ */
