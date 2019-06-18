#ifndef MV_WORKLOADS
#define MV_WORKLOADS

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/target/keembay/rectangle.hpp"
#include "include/mcm/target/keembay/workload_struct.hpp"
#include <metis.h>
#include <climits>
#include <math.h>
#include <cmath>
#include <vector>
#include <numeric>

namespace mv
{
    /** 
    * @brief Cost Function types to be used when evaluating execution cycles of a workload 
    */ 
    enum class CostFunctions
    {
        Balanced,
        CriticalPath,
        Greedy,
        MinMaxWorkloads
    };

    struct DPUMode { unsigned H, W; };
    using  DPUModeList = std::vector<mv::DPUMode>;

    /* The POC compiler generates a lattic structure of the tensor shape with the nodes numbered in this order
     * Example for tensor size 16x16
     * 
     *         axis numbering
     *     
     *         0    4     8    12     
     *        
     *    0    0----2-----4----6 //Even numbers
     *         |    |     |    | 
     *    4    1----3-----5----7 //Odd numbers
     *         |    |     |    | 
     *    8    10---11----12---13 // Incrementing numbers
     *         |    |     |    | 
     *    12   15---16----17---18
     */

    struct MetisGraphStructure
    {
        std::unique_ptr<idx_t[]>  xadj;   /*Indexes of starting points in adjacent array*/
        std::unique_ptr<idx_t[]>  adjncy; /*Adjacent vertices in consecutive index order*/
        std::unique_ptr<idx_t[]>  part;
        std::unique_ptr<idx_t[]>  vwgt;

        idx_t objval;
        idx_t nWeights  = 1;              /*Each vertex stores 1 weight*/
        idx_t options[METIS_NOPTIONS];

        idx_t m_numberTensorVertices;
        idx_t m_numberTensorEdges;
        int m_xDim;
        int m_yDim;
        int n_elem_y;
        int n_elem_x;
        double tensorXDim;
        double tensorYDim;

        std::unique_ptr<mv::Rectangle[]>  node_coords;

        MetisGraphStructure(mv::Shape outputTensor, mv::DPUMode MPEMode);
    };
  
    struct point
    {
        int16_t x = 0;
        int16_t y = 0;
    };

    enum class WorkloadSplitMode { HW=1, HC, WC };

    class Workloads : public LogSender
    {

        std::vector<Workload> workloads_;
        std::string layerName_;
        mv::Shape tensorShape_;
        std::vector<float> executionCycles_; //Min & Max execution cycles
        float meanExecutionCycles_ = 0;

        std::shared_ptr<MetisGraphStructure> metisGraph_;

        float critical_workload = 0;
        float workload_sum = 0;
        float min_range = 0;
        float max_range = 0;

        std::vector<int> generateMetisGraphNodeNumbers(void);

    public:
        Workloads(const std::string& name, const mv::Shape& tensorShape);
        Workloads(const std::string& name, const mv::Shape& tensorShape, mv::DPUMode& mpeMode);
        ~Workloads();
      
        void generateMetisGraph(void);
        std::shared_ptr<mv::MetisGraphStructure> getMetisGraph();

        /*returns: METIS_OK(=1), or METIS_ERROR*/
        int partitionTensorWithRectangleHeuristic(const mv::DPUModeList& modes,
                                                            idx_t        nWorkloads,
                                                            bool         split_over_h,
                                                            bool         split_over_w,
                                                            bool         split_symmetric,
                                                  const mv::WorkloadSplitMode& split_mode,
                                                  const mv::pass::PassEntry& pass);

        idx_t getNWorkloads(const mv::Shape& tensorShape, int nDPUxCluster);

        void populateWorkloadsFromPartitions(idx_t nWorkloads, 
                                            const mv::pass::PassEntry& pass, 
                                            mv::DPUMode& mpeMode);

        std::vector<mv::Workload> polygonWorkloadSplit(const mv::pass::PassEntry& pass, 
                                                        mv::Workload& workload, 
                                                        std::vector<mv::Workload>& workloads, 
                                                        mv::DPUMode& mpeMode);
        
        std::vector<mv::Workload> workloadSplitHelper(const mv::pass::PassEntry& pass, 
                                                        mv::Workload& workload, 
                                                        std::pair<std::pair<int16_t, int16_t>,bool>& interesting_point, 
                                                        mv::DPUMode& mpeMode);
        std::size_t nWorkloads() const;
        void addWorkload(mv::Workload workload);
        const std::vector<mv::Workload>& getWorkloads() const;
        static const std::vector<int> getWorkloadSplitPool(const Tensor& tensor, int nDPUxCluster, mv::DPUModeList dpuModeList, int maxSplits);

        static void generateExecutionCycles(std::vector<mv::Workloads>& workloadsVector, int nDPUxCluster, CostFunctions costFunction);
        std::vector<float> getExecutionCycles() const;
        float getMeanExecutionCycles() const;
        void setExecutionCycles(std::vector<float> val);
        static float greedyTaskAssignment(int nProcessors, std::vector<float>& workloadCosts);

        bool validateWorkloads(std::vector<mv::Data::TensorIterator>& inputTensor);
        bool validateWorkloads(const mv::Shape& shape);

        static mv::CostFunctions getCostFunction(mv::Element& passDesc, const mv::pass::PassEntry& pass);

        double getAllWorkloadsVolume() const;
        bool noOverlap() const;
        mv::Shape getShapefromMinMax() const;

        Workload& operator[](int nworkload);
        bool operator < (const mv::Workloads& other) const;
        
        const Workload& operator[](int nworkload) const;
        std::string getLogID() const override;
        std::string toString() const;
        std::string toLongString() const;
    };
}

#endif 
