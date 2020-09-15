#ifndef _IMPLICIT_FLOW_HPP_
#define _IMPLICIT_FLOW_HPP_

#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/graph/graph.hpp"

namespace mv
{
    class ImplicitFlow
    {
    public:
        enum Direction
        {
            INPUT_IN_OUTPUT = 0,
            OUTPUT_IN_INPUT = 1,
            MIXED = 3,
            UNSPECIFIED = 4
        };
    private:
        bool candidate_;
        bool resolved_;
        bool passtrough_;
        const Direction compensationDirection_;
    public:
        ImplicitFlow() :
            candidate_(true),
            resolved_(false),
            passtrough_(false),
            compensationDirection_(UNSPECIFIED)
        {};

        ImplicitFlow(bool candidate) :
            candidate_(candidate),
            resolved_(false),
            passtrough_(false),
            compensationDirection_(UNSPECIFIED)
        {};

        ImplicitFlow(Direction direction) :
            candidate_(true),
            resolved_(false),
            passtrough_(false),
            compensationDirection_(direction)
        {};

        ImplicitFlow(Direction direction, bool candidate) :
            candidate_(candidate),
            resolved_(false),
            passtrough_(false),
            compensationDirection_(direction)
        {};

        void resolve()
        {
            //todo: ask around about mcmCompiler error exceptions
            if(candidate_)
            {
                resolved_ = true;
            }
            else
            {
                //todo: throw some exception or something
            }
        }

        void setPasstrough() {
            resolve();
            passtrough_ = true;
        }

        void appoint(bool appointment)
        {
            candidate_ = appointment;
            resolved_ = false;
            passtrough_ = false;
        }

        bool isImplicit()
        {
            return (resolved_ or passtrough_ );
        }

        bool isPasstrough()
        {
            return (passtrough_);
        }

        bool isCandidate()
        {
            return candidate_;
        }

        Direction getCompensationDirection()
        {
            return compensationDirection_;
        }
    };
}

#endif // _IMPLICIT_FLOW_HPP_
