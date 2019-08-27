#include "include/mcm/target/keembay/workloads.hpp"

mv::MetisGraphStructure::MetisGraphStructure(mv::Shape outputTensor, mv::DPUMode MPEMode) {
    
    /*Shape of output tensor x-y*/
    tensorXDim = outputTensor[0]; 
    tensorYDim = outputTensor[1];

    /*Number of vertices and edges in METIS lattic graph of tensor*/
    m_numberTensorVertices = ceil(tensorXDim / MPEMode.H)  * ceil(tensorYDim / MPEMode.W);
    m_numberTensorEdges = (2 * ceil(tensorXDim / MPEMode.H) * ceil(tensorYDim / MPEMode.W)) - ceil(tensorXDim / MPEMode.H) - ceil(tensorYDim / MPEMode.W);
        
    /*X-Y dimension of METIS lattic graph*/
    m_xDim = ceil((tensorXDim / MPEMode.W));
    m_yDim = ceil((tensorYDim / MPEMode.H));

    /*METIS parameters - description page 23 Metis manual*/
    xadj.reset(new idx_t[m_numberTensorVertices + 1]);
    adjncy.reset(new idx_t[2*m_numberTensorEdges]);
    part.reset(new idx_t[m_numberTensorVertices]);
    vwgt.reset(new idx_t[m_numberTensorVertices* nWeights]);

    node_coords.reset(new mv::Rectangle [m_numberTensorVertices]);
        
    /* (1) This section gnerates weights for the METIS vertices
    * Description page 23 Metis manual
    * 
    * This is required when the tensor dimensions are not a multiple of 4 for MPE mode (4,4) or 16 for MPE mode (1,16)
    * When tensor size is not a multiple of the MPE dimensions a full DPUs will be fully utilised (i.e. < 256 multiplication operations)
    * Therefore we assign nodes different weights when partitioning
    * 
    * (2) We populate (x,y) coordinates for the individual lattic nodes here with the rectangle class. 
    * 
    */

    int nodeIndex = 0; /* This corresponds to the numbering format in the lattic structure*/

    /* We need to handle the first two rows of the lattic first, see node numbering in the lattic example above*/
    /* Here we populate the the coordiantes of the nodes in the lattic*/
    /* We need to handle the first two rows of the lattic first, see node numbering in the lattic example above*/
    for(int j=0; j < 1; j++) {
        
        if ((j+1 < m_yDim) || (!fmod(tensorYDim,MPEMode.H)))
            n_elem_y = MPEMode.H;                 
        else 
            n_elem_y = (int)tensorYDim%MPEMode.H; 
                
        /*This loops over the the first two rows 1,2,3,4 .... etc*/
        for(int k=0; k < (m_xDim*2); k++) {

            int min_x;
            int min_y;
                    
            if((k%2 != 0) && (m_yDim <= 2)) 
                n_elem_y = (int)tensorYDim%MPEMode.H;
            else
                n_elem_y = MPEMode.H; 
                    
            if ((k < (m_xDim*2)-2) || (!fmod(tensorXDim,MPEMode.W)))
                n_elem_x = MPEMode.W;
            else 
                n_elem_x = (int)tensorXDim%MPEMode.W;
                    
            /*First row where node number is even i.e. 2,4,6... */
            if ((nodeIndex%2 == 0) && (nodeIndex <= ((m_xDim*2)-2))) { 

                min_x = (k/2) * MPEMode.W;
                min_y = j * MPEMode.H;

                assert(nodeIndex < m_numberTensorVertices);
                node_coords[nodeIndex] = mv::Rectangle(min_x, min_y, n_elem_x, n_elem_y);

                assert(nodeIndex < m_numberTensorVertices * nWeights);
                vwgt[nodeIndex] = n_elem_x * n_elem_y; /* Populate METIS weight*/
            }
            
            /*Second row where node number is odd i.e. 1,3,5... */
            if ((nodeIndex%2 != 0) && (nodeIndex <= ((m_xDim*2)-1))) {
                        
                /*For 7x7 tensor mode 4x4*/
                if(m_yDim <= 2) {
                    min_x = min_x;
                    min_y = min_y + n_elem_y + 1;
                }
                else {
                    min_x = min_x;
                    min_y = min_y + n_elem_y;
                }
                        
                assert(nodeIndex < m_numberTensorVertices);
                node_coords[nodeIndex] = mv::Rectangle(min_x, min_y, n_elem_x, n_elem_y);

                assert(nodeIndex < m_numberTensorVertices * nWeights);
                vwgt[nodeIndex] = n_elem_x * n_elem_y; /* Populate METIS weight*/
            }        
            nodeIndex++;
        }
    }
    
    /* Now deal with the remaining rows after the first 2 rows*/
    for(int j=2; j < m_yDim; j++) { 
            
        if ((j+1 < m_yDim) || (!fmod(tensorYDim,MPEMode.H)))
            n_elem_y = MPEMode.H;                 
        else 
            n_elem_y = (int)tensorYDim%MPEMode.H; 
                            
        for(int k=0; k < m_xDim; k++) {

            if ((k+1 < m_xDim) || (!fmod(tensorXDim,MPEMode.W)))
                n_elem_x = MPEMode.W;
            else 
                n_elem_x = (int)tensorXDim%MPEMode.W;

            assert(nodeIndex < m_numberTensorVertices * nWeights);
            vwgt[nodeIndex] = n_elem_x * n_elem_y; /* Populate METIS weight*/

            int min_x = k * MPEMode.W;
            int min_y = j * MPEMode.H;

            assert(nodeIndex < m_numberTensorVertices);
            node_coords[nodeIndex] = mv::Rectangle(min_x, min_y, n_elem_x, n_elem_y);
                 
            nodeIndex++;
        }
    }
}