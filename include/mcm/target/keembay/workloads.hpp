#ifndef MV_WORKLOADS
#define MV_WORKLOADS

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/target/keembay/rectangle.hpp"
#include <metis.h>
#include <climits>
#include <math.h>

namespace mv
{
    enum MPE_Mode
    {
        Vector,
        Matrix
    };

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

    /* METIS parameters*/
    struct MetisGraphStructure
    {
        idx_t* xadj;                      /*Indexes of starting points in adjacent array*/
        idx_t* adjncy;                    /*Adjacent vertices in consecutive index order*/
        idx_t* vwgt;
        idx_t* part; 
        idx_t objval;
        idx_t nWeights  = 1;              /*Each vertex stores 1 weight*/
        idx_t options[METIS_NOPTIONS];

        idx_t m_numberTensorVertices;
        idx_t m_numberTensorEdges;
        int m_xDim;
        int m_yDim;

        mv::Rectangle* node_coords;

        MetisGraphStructure(mv::Shape outputTensor, std::pair <int,int> MPEMode) {

            /*Shape of output tensor x-y*/
            double tensorXDim = outputTensor[0]; 
            double tensorYDim = outputTensor[1];

            /*Number of vertices and edges in METIS lattic graph of tensor*/
            m_numberTensorVertices = ceil(tensorXDim / MPEMode.first)  * ceil(tensorYDim / MPEMode.second);    
            m_numberTensorEdges = (2 * ceil(tensorXDim / MPEMode.first) * ceil(tensorYDim / MPEMode.second)) - ceil(tensorXDim / MPEMode.first) - ceil(tensorYDim / MPEMode.second);
        
            /*X-Y dimension of METIS lattic graph*/
            m_xDim = ceil((tensorXDim / MPEMode.first));
            m_yDim = ceil((tensorYDim / MPEMode.second));

            /*METIS parameters - description page 23 Metis manual*/
            xadj = new idx_t[m_numberTensorVertices + 1]; 
            adjncy = new idx_t[2*m_numberTensorEdges];
            part = new idx_t[m_numberTensorVertices];
            vwgt = new idx_t[m_numberTensorVertices* nWeights];

            node_coords = new mv::Rectangle [ m_numberTensorVertices ];
        
            /* Weights of METIS vertices
             * Description page 23 Metis manual
             * 
             * Required when tensor size is not a multiple of 4 for MPE mode (4,4) which is only supported for WW09
             * When tensor size is not a multiple of 4 then not all DPUs will be fully utilised (i.e. < 256 multiplication operations)
             * Therefore we assign nodes different weights when partitioning
            */
            int n_elem_y;
            int n_elem_x;
            int nodeIndex = 0;
            for(int j=0; j < m_yDim; j++) {
            
                if ((j+1 < m_yDim) || (!fmod(tensorYDim,MPEMode.first)))
                    n_elem_y = MPEMode.first;                 
                else 
                    n_elem_y = (int)tensorYDim%MPEMode.first; 
                            
                for(int k=0; k < m_xDim; k++) {
                
                    if ((k+1 < m_xDim) || (!fmod(tensorXDim,MPEMode.second)))
                        n_elem_x = MPEMode.second;
                    else 
                        n_elem_x = (int)tensorXDim%MPEMode.second;
            
                    vwgt[nodeIndex] = n_elem_x * n_elem_y;

                    int min_x = k * MPEMode.first;
                    int min_y = j * MPEMode.second;
        
                    node_coords[nodeIndex] = mv::Rectangle(min_x, min_y, n_elem_x, n_elem_y);
                    nodeIndex++;
                }
            }
        }   
    
        ~MetisGraphStructure() {
            delete[] xadj;
            delete[] adjncy;
            delete[] part;
            delete[] vwgt;
            delete[] node_coords;
        }
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
        MetisGraphStructure *metisGraph_;
     
    public:
        Workloads(const std::string& name, const mv::Shape& tensorShape, std::pair <int,int>& mpeMode);
        ~Workloads();
        MetisGraphStructure& getMetisGraph(); 
        void generateMetisGraph(MetisGraphStructure& metisGraph);
        int partitionTensorMETIS(MetisGraphStructure& metisGraph, idx_t nWorkloads);
        idx_t getNWorkloads(const mv::Shape& tensorShape, int nDPUxCluster);

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
