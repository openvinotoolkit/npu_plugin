#ifndef MV_WORKLOADS
#define MV_WORKLOADS

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/tensor/shape.hpp"

namespace mv
{
    enum MPE_Mode
    {
        Vector,
        Matrix
    };

    struct Workload
    {
        MPE_Mode MPEMode;
        int16_t MaxX = 0;
        int16_t MaxY = 0;
        int16_t MaxZ = 0;
        int16_t MinX = 0;
        int16_t MinY = 0;
        int16_t MinZ = 0;
        int16_t padLeft = 0; //Are workload paddings different from full tensor padding?
        int16_t padRight = 0;
        int16_t padTop = 0;
        int16_t padBottom = 0;
        int32_t clusterID = 0;
        int8_t workloadID = 0;

    };
    
    class Workloads : public LogSender 
    {

        std::vector<Workload> workloads_;
        std::string layerName;
     
    public:
        Workloads(const std::string& name);
        std::size_t nWorkloads() const;
        std::vector<Workload>& getWorkloads(); 
        Workload& operator[](int nworkload);
        const Workload& operator[](int nworkload) const;
        std::string getLogID() const override;
        std::string toString() const;
        double getAllWorkloadsVolume() const;
        bool noOverlap() const;
        mv::Shape getShapefromMinMax() const;
    };
}

#endif 
