/*#ifndef INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_
#define INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_
=======
#ifndef INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_
#define INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_
>>>>>>> master

#include "include/mcm/api/compositional_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/graph/graph.hpp"
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
		// Recorded source file
		std::ofstream outputSourceFile;
		// Recorded weights file
		std::ofstream outputWeightsFile;
		// Stream for source file
		std::ostringstream ss;
		// Stream for weights file
		std::ostringstream ws;
		std::string recordedSourceFileName;
		std::string recordedWeghtsFileName;
		std::string recordedSourceFileNameCpp;
		const std::string savedRecordingsPath_;
		unsigned weightsVectorCounter = 0;
		unsigned concatVectorCounter = 0;

		static std::string toString(const std::array<unsigned short, 2>& arr);
		static std::string toString(const std::array<unsigned short, 4>& arr);

	public:

        CompositionalModelRecorder(OpModel& model, std::string recordingsPath_);
        ~CompositionalModelRecorder();
        virtual Data::TensorIterator input(const Shape& shape, DType dType, Order order, const std::string& name = "") override;
		virtual Data::TensorIterator output(Data::TensorIterator input, const std::string& name = "") override;
        virtual Data::TensorIterator constant(const std::vector<double>& data, const Shape& shape, DType dType, Order order, const std::string& name = "") override;
		virtual Data::TensorIterator conv2D(Data::TensorIterator input, Data::TensorIterator filters, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
        virtual Data::TensorIterator depthwiseConv2D(Data::TensorIterator input, Data::TensorIterator filters, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
		virtual Data::TensorIterator matMul(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
		virtual Data::TensorIterator maxpool2D(Data::TensorIterator input, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
		virtual Data::TensorIterator avgpool2D(Data::TensorIterator input, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") override;
		virtual Data::TensorIterator concat(std::vector<mv::Data::TensorIterator>& inputs, const std::string& name = "") override;
		virtual Data::TensorIterator batchNorm(Data::TensorIterator input, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, double varianceEps, const std::string& name = "") override;
		virtual Data::TensorIterator scale(Data::TensorIterator input, Data::TensorIterator scale, const std::string& name = "") override;
		virtual Data::TensorIterator relu(Data::TensorIterator input, const std::string& name = "") override;
		virtual Data::TensorIterator prelu(Data::TensorIterator input, Data::TensorIterator negativeSlope, const std::string& name = "") override;
		virtual Data::TensorIterator softmax(Data::TensorIterator input, const std::string& name = "") override;
		virtual Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
		virtual Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
		virtual Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
		virtual Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, const std::string& name = "") override;
		virtual Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, const std::string& name = "") override;
		virtual Data::TensorIterator bias(Data::TensorIterator input, Data::TensorIterator biases, const std::string& name = "") override;
		virtual Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, const std::string& name = "") override;
		// Create two files to store the generated source code and weight vector definitions
		void createRecordedSourceFiles();
		// Populate the 'compilation passes' and end of the source file 
		void completeRecordedSourceFile();

		void writeWeightsToFile(const std::vector<double>& weightsData, std::string weightsVectorName);

		bool isValid() const override;
		bool isValid(const Data::TensorIterator& it) const override;
		bool isValid(const Data::OpListIterator& it) const override;

		Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;
		void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) override;

		std::string getLogID() const override;

	};

}
#endif //  INCLUDE_MCM_UTILS_COMPOSITIONAL_MODEL_RECORDER_HPP_ */
