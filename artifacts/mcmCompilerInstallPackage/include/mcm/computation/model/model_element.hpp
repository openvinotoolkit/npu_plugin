#ifndef MV_MODEL_ELEMENT
#define MV_MODEL_ELEMENT

#include <string>
#include <array>
#include <functional>
#include "include/mcm/base/element.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"

namespace mv
{

    class ComputationModel;

    class ModelElement : public Element
    {

        std::reference_wrapper<ComputationModel> model_;

    protected:

        ComputationModel& getModel_();

    public:

        ModelElement(ComputationModel& model, const std::string& name);
        virtual ~ModelElement() = 0;

        virtual json::Value toJSON() const override;
        virtual std::string getLogID() const override;

    };

}

#endif // COMPUTATION_OP_HPP_