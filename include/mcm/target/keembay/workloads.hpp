#ifndef MV_WORKLOADS
#define MV_WORKLOADS

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"

namespace mv
{
    struct Workload
    {
        std::pair <int,int> MPEMode;
        int16_t MaxX;
        int16_t MaxY;
        int16_t MaxZ;
        int16_t MinX;
        int16_t MinY;
        int16_t MinZ;
        int16_t padLeft;
        int16_t padRight;
        int16_t padTop;
        int16_t padBottom;
        int32_t clusterID;
        int8_t workloadID;

    };
    
    class Workloads : public LogSender // : public ModelElement
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
        
    };
}

#endif 