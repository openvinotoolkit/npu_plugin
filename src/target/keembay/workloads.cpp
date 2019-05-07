#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include <algorithm> 
#include <metis.h>
#include <sstream>
mv::Workloads::Workloads(const std::string& name, const mv::Shape& tensorShape, std::pair <idx_t,idx_t>& mpeMode):
layerName_(name), tensorShape_(tensorShape), metisGraph_(new MetisGraphStructure(tensorShape, mpeMode))
{
    
}

mv::Workloads::~Workloads()
{
}

mv::Workload& mv::Workloads::operator[](int nworkload)
{
    return const_cast<Workload&>(static_cast<const Workloads*>(this)->operator[](nworkload));
}

bool mv::Workloads::operator < (const mv::Workloads& other) const
{
    /* Sort the workloads based on mean Execution cycles then workloads count */
    
    //mean of higher/lower execution cycles
    float lhs_avg = (executionCycles_[0] + executionCycles_[1]) / 2;
    float rhs_avg = (other.getExecutionCycles()[0] + other.getExecutionCycles()[1]) / 2;

    if (lhs_avg == rhs_avg)
        return executionCycles_.size() < other.getWorkloads().size();
    else
        return lhs_avg < rhs_avg;
}


const mv::Workload& mv::Workloads::operator[](int nworkload) const
{

    if (nworkload >= static_cast<int>(workloads_.size()) || static_cast<int>(workloads_.size()) + nworkload < 0)
        throw ArgumentError(*this, "index subscript", std::to_string(nworkload),
            "Exceeds the dimensionality " + std::to_string(nWorkloads()));

    if (nworkload < 0)
        return workloads_[workloads_.size() + nworkload];

    return workloads_[nworkload];

}

std::size_t mv::Workloads::nWorkloads() const
{
    return workloads_.size();
}

const std::vector<mv::Workload>& mv::Workloads::getWorkloads() const
{
    return workloads_;
}

std::string mv::Workloads::toString() const
{
    std::string output = "{";
    
    for (std::size_t i = 0; i < this->nWorkloads(); ++i) {
        output += "MinX " + std::to_string(this->workloads_[i].MinX) + ", ";
        output += "MaxX " + std::to_string(this->workloads_[i].MaxX) + ", ";
        output += "MinY " + std::to_string(this->workloads_[i].MinY) + ", ";
        output += "MaxY " + std::to_string(this->workloads_[i].MaxY) + ", ";
        output += "MinZ " + std::to_string(this->workloads_[i].MinZ) + ", ";
        output += "MaxZ " + std::to_string(this->workloads_[i].MaxZ) + ", ";
        output += "MaxZ " + std::to_string(this->workloads_[i].MaxZ) + ", ";
        output += "WorkloadID " + std::to_string(this->workloads_[i].workloadID) + ", ";
        output += "ClusterID " + std::to_string(this->workloads_[i].clusterID) + ", ";
        }
        output += "}";

        return output;
}

std::string mv::Workloads::getLogID() const
{
    return "Workloads:" + toString();
}

double mv::Workloads::getAllWorkloadsVolume() const
{
    double volume = 0;
    for (std::size_t i = 0; i < this->nWorkloads(); ++i)
    {
        std::int16_t volX = this->workloads_[i].MaxX - this->workloads_[i].MinX;
        std::int16_t volY = this->workloads_[i].MaxY - this->workloads_[i].MinY;
        std::int16_t volZ = this->workloads_[i].MaxZ - this->workloads_[i].MinZ;
        //volume += (this->workloads_[i].MaxX - this->workloads_[i].MinX + 1) * (this->workloads_[i].MaxY - this->workloads_[i].MinY + 1) * (this->workloads_[i].MaxZ - this->workloads_[i].MinZ + 1);
        volume += std::max(volX, static_cast<std::int16_t>(1)) * std::max(volY, static_cast<std::int16_t>(1)) * std::max(volZ, static_cast<std::int16_t>(1));
    }
    return volume;
}

bool mv::Workloads::noOverlap() const
{
    bool noIntersect = true;
    for(std::size_t i=0; i< this->nWorkloads(); i++)
        {
            for(std::size_t j=0;j<this->nWorkloads();j++){
                if(i==j)
                {
                    continue;
                }
        // applying De Morgan's law ((A U B)' == A' ): Two rectangles donot overlap if one rectangle's minimum in a dimension is greater than the other rectangle's maximum in that dimension
        // check to be done for both the X and Y dimension.
                noIntersect = noIntersect &&
                             (this->workloads_[i].MinX > this->workloads_[j].MaxX ||
                              this->workloads_[j].MinX > this->workloads_[i].MaxX ||
                              this->workloads_[i].MinY > this->workloads_[j].MaxY ||
                              this->workloads_[j].MinY > this->workloads_[i].MaxY ||
                              this->workloads_[i].MinZ > this->workloads_[j].MaxZ ||
                              this->workloads_[j].MinZ > this->workloads_[i].MaxZ);
            
            }

        }
        return noIntersect; 
}

mv::Shape mv::Workloads::getShapefromMinMax() const
{
    // get the global min and max of the workloads
    std::int16_t minX=0, maxX=0, minY=0, maxY=0, minZ=0, maxZ = 0;
    for(std::size_t i=0; i< this->nWorkloads(); i++)
    {
        minX = minX < this->workloads_[i].MinX ? minX : this->workloads_[i].MinX;
        minY = minY < this->workloads_[i].MinY ? minY : this->workloads_[i].MinY;
        minZ = minZ < this->workloads_[i].MinZ ? minZ : this->workloads_[i].MinZ;
        maxX = maxX > this->workloads_[i].MaxX ? maxX : this->workloads_[i].MaxX;
        maxY = maxY > this->workloads_[i].MaxY ? maxY : this->workloads_[i].MaxY;
        maxZ = maxZ > this->workloads_[i].MaxZ ? maxZ : this->workloads_[i].MaxZ;
    }

    std::vector<std::size_t> minMax;
    minMax.push_back(maxX - minX);
    minMax.push_back(maxY - minY);
    minMax.push_back(maxZ - minZ);
    return mv::Shape(minMax);
}

/*
 * @brief Generates a vector of node numbers to be used to create the METIS adjacency structure.
 *        The sequence of nodes is per graph below. Note the order of the first 2 two rows. 
 * @return A vector of node numbers 
 */

/* The POC compiler generates a lattic structure of the tensor shape with the nodes numbered in this order
   * Example for tensor size 16x16
   * 
   *   0----2-----4----6------8
   *   |    |     |    |      |
   *   1----3-----5----7------9
   *   |    |     |    |      |
   *   10---11----12---13----14
   *   |    |     |    |      |
   *   15---16----17---18----19
   */
std::vector<int> mv::Workloads::generateMetisGraphNodeNumbers(void) {

    /*Generate sequence of node numberes for the lattic structure of the tensor shape*/
    std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph_->m_numberTensorVertices);

    for(int i = 1; i < metisGraph_->m_xDim; i++) {
        nodeNumbers[i] = nodeNumbers[i-1] + 2; 
    }

    nodeNumbers[metisGraph_->m_xDim] = 1;
    for(int k = metisGraph_->m_xDim + 1; k < (metisGraph_->m_xDim * 2); k++) {
        nodeNumbers[k] = nodeNumbers[k-1] + 2; 
    }

    return nodeNumbers;
}

/*
 * @brief Creates a METIS adjacency structure of a graph as per 23/45 METIS manual. 
 * @brief Representing the lattic structure of the tensor shape (in the X-Y corrdinate) 
 * @param metisGraph - a struct containing necessary parameters to pass to METIS
 * @return None
 * 
 */

 /* The POC compiler generates a lattic structure of the tensor shape with the nodes numbered in this order
    McM compiler impliments the same numbering approach to ensure correctness
   * Example for tensor size 16x16
   * 
     * 0----2-----4----6-----8
     * |    |     |    |     |
     * 1----3-----5----7-----9
     * |    |     |    |     |
     * 10---11----12---13---14
     * |    |     |    |     |
     * 15---16----17---18---19
     */

void mv::Workloads::generateMetisGraph(void) {


    /*If the lattice structure has more than 1 column the node numbering will be like this the lattic graph above (first row even, second row odd)*/

    if((metisGraph_->m_xDim > 1) && (metisGraph_->m_yDim > 1)) {
    
        /*Generate sequence of node numberes for the lattic structure of the tensor shape*/
        auto nodeNumbers  = this->generateMetisGraphNodeNumbers();

        int adjncyIndex = 0;
        int xadjIndex = 0;
        int increment = 1;

        /* The first two rows of the lattic structure have a different order to the remaining rows (see graph).
        * There we need to populate the adjancy structures for the first two rows first seperately.
        */

        for (std::vector<int>::iterator it = nodeNumbers.begin(); it != (nodeNumbers.begin()+(metisGraph_->m_xDim * 2)); std::advance(it,increment)) {

            /*Top left node, i.e. 0*/ 
            if((*it%metisGraph_->m_xDim == 0) && (*it == 0)) {

                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + metisGraph_->m_xDim]; 
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + 1];
                adjncyIndex++;
        }
   
        /*Top right node, i.e 8 in the example graph*/
        if(*it == ((metisGraph_->m_xDim * 2) - 2)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 2;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1; 
            adjncyIndex++;
        }

        /*Middle nodes of the top row*/
        if((*it != 0) && (*it < ((metisGraph_->m_xDim * 2)-2)) && (*it%2 == 0)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 2;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 2; 
            adjncyIndex++;
        }

        /*Middle nodes of the second row*/
        if((*it != 0) && (*it != 1) && (*it < (metisGraph_->m_xDim * 2)) && (*it%2 != 0) && (*it != (metisGraph_->m_xDim * 2)-1)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 2; 
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = nodeNumbers[std::distance(nodeNumbers.begin(), it) + metisGraph_->m_xDim];
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 2; 
            adjncyIndex++;
        }
        
        /*second row first node i.e. 1*/
        if((*it == 1)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = nodeNumbers[(metisGraph_->m_xDim * 2)];
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 2; 
            adjncyIndex++;
        }

        /*second row last node on right side*/
        if((*it == (metisGraph_->m_xDim * 2)-1)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 2;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = nodeNumbers[std::distance(nodeNumbers.begin(), it) + metisGraph_->m_xDim];
            adjncyIndex++;
        }

        /*Depending whether we are on the first or second row then we increment by a different amount*/
        /*First row*/
        if(*it%2 == 0) {
            increment = metisGraph_->m_xDim;
        }
        /*Second row*/
        else {
            increment = metisGraph_->m_xDim-1;
            increment = -increment;
        }
        /*If on the last node of the second row then we done, break*/ 
        if(*it == (metisGraph_->m_xDim * 2)-1)
            break;
        }

        /* The 3rd and ramining rows of the lattic structure have a different order to the first two rows (see graph).
        * There we need to populate the adjancy structures for these rows first seperately.
        */
        for (std::vector<int>::iterator it = (nodeNumbers.begin()+(metisGraph_->m_xDim * 2)); it != nodeNumbers.end(); it++) {

            /*Intermediate node left side*/ 
            if((*it%metisGraph_->m_xDim == 0) && ((*it + metisGraph_->m_xDim) < ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - metisGraph_->m_xDim];
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + metisGraph_->m_xDim]; 
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + 1];
                adjncyIndex++;
            }

            /*Bottom left node*/
            if((*it%metisGraph_->m_xDim == 0) && ((*it + metisGraph_->m_xDim) > ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - metisGraph_->m_xDim];
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + 1];
                adjncyIndex++;
            }

            /*Intermediate right side node*/
            if(((*it - (metisGraph_->m_xDim-1))%metisGraph_->m_xDim == 0) && ((*it - (*it -(metisGraph_->m_xDim-1))) == metisGraph_->m_xDim -1)  && ((*it-(metisGraph_->m_xDim-1) != 0))  && (*it %(nodeNumbers.size()-1) != 0)) {

                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - metisGraph_->m_xDim];
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - 1];
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + metisGraph_->m_xDim]; 
                adjncyIndex++;
            }

            /*Bottm right node*/
            if(((*it - (metisGraph_->m_xDim-1))%metisGraph_->m_xDim == 0) && ((*it - (*it -(metisGraph_->m_xDim-1))) == metisGraph_->m_xDim -1) && (*it %(nodeNumbers.size()-1) == 0)) {

                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - metisGraph_->m_xDim]; 
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - 1];
                adjncyIndex++;
                metisGraph_->xadj[xadjIndex] = adjncyIndex;
            }
        
            /*Middle nodes bottom row*/
            if(((*it)%metisGraph_->m_xDim != 0) && ((*it) > ((int)nodeNumbers.size()-1) - metisGraph_->m_xDim) && ((*it) != ((int)nodeNumbers.size()-1))) {

                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - metisGraph_->m_xDim]; 
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - 1];
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + 1];
                adjncyIndex++;
            }

            /*Middle nodes not on bottom or top rows or the side columns*/
            if(((*it)%metisGraph_->m_xDim != 0) && ((*it) < ((int)nodeNumbers.size()-1) - metisGraph_->m_xDim) && ((*it) > (metisGraph_->m_xDim-1)) && ((*it+1)%metisGraph_->m_xDim != 0)) {

                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - metisGraph_->m_xDim]; 
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it - 1];
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + metisGraph_->m_xDim]; 
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = nodeNumbers[*it + 1];
                adjncyIndex++;
            }
        }
    }
    /*There is only one column in the lattic and node are numbered in order like this*/
    /*
     * 0
     * | 
     * 1
     * |    
     * 2
     * |   
     * 3
     */
    else {
        
        /*Nodes in the graph*/
        std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph_->m_numberTensorVertices);
        int adjncyIndex = 0;
        int xadjIndex = 0;

        for (std::vector<int>::iterator it = nodeNumbers.begin(); it != nodeNumbers.end(); it++) {

            if((*it) == 0) {
                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = 1;
                adjncyIndex++;
            }
            if((*it) == metisGraph_->m_yDim-1) {
                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = metisGraph_->m_yDim-2; 
                adjncyIndex++;
                metisGraph_->xadj[xadjIndex] = adjncyIndex;
            }
            if(((*it) > 0) && ((*it) < metisGraph_->m_yDim-1)) {
                metisGraph_->xadj[xadjIndex] = adjncyIndex;
                xadjIndex++;
                metisGraph_->adjncy[adjncyIndex] = *it -1; 
                adjncyIndex++;
                metisGraph_->adjncy[adjncyIndex] = *it + 1; 
                adjncyIndex++;

            
            }

        }
    }
} 

int mv::Workloads::partitionTensorWithMETIS(idx_t nWorkloads, const mv::pass::PassEntry& pass) 
{
    METIS_SetDefaultOptions(metisGraph_->options);

    pass.log(mv::Logger::MessageType::Debug, "The adjancy data for METIS is ");
    for(int i =0; i < 2*metisGraph_->m_numberTensorEdges; i++) 
        pass.log(mv::Logger::MessageType::Debug, std::to_string(metisGraph_->adjncy[i]));
    
    pass.log(mv::Logger::MessageType::Debug, "The xadj data for METIS is ");
    for(int i =0; i < (metisGraph_->m_numberTensorVertices + 1); i++) 
        pass.log(mv::Logger::MessageType::Debug, std::to_string(metisGraph_->xadj[i]));
    
    pass.log(mv::Logger::MessageType::Debug, "The vwgt data for METIS is ");
    for(int i =0; i < (metisGraph_->m_numberTensorVertices); i++) 
        pass.log(mv::Logger::MessageType::Debug, std::to_string(metisGraph_->vwgt[i]));
    
    /*METIS call*/
    int res = METIS_PartGraphRecursive(&metisGraph_->m_numberTensorVertices,&metisGraph_->nWeights, metisGraph_->xadj.get(), metisGraph_->adjncy.get(),
                    metisGraph_->vwgt.get(), NULL, NULL, &nWorkloads, NULL,
				    NULL, metisGraph_->options, &metisGraph_->objval, metisGraph_->part.get());

    pass.log(mv::Logger::MessageType::Debug, "Value of the objective function that was minimized by METIS (should be same as PoC compiler) is: " + std::to_string(metisGraph_->objval));

    /*Print node partition*/
    for(int part_i = 0; part_i < metisGraph_->m_numberTensorVertices; part_i++) 
            pass.log(mv::Logger::MessageType::Debug, "Node " + std::to_string(part_i) + " is in partition " + std::to_string(metisGraph_->part[part_i]));  
    
    return res;
}

/*TODO update*/
idx_t mv::Workloads::getNWorkloads(const mv::Shape& tensorShape, int nDPUxCluster) {
    
    return round(nDPUxCluster/2)*2; 
}


void mv::Workloads::populateWorkloadsFromPartitions(idx_t nWorkloads, const mv::pass::PassEntry& pass, std::pair <idx_t,idx_t>& mpeMode) 
{
    std::vector<std::vector<mv::Workload>> listOfworkloadLists;
                
    for(int workload = 0; workload < nWorkloads; workload++) { 
        
        workloads_.push_back(mv::Workload()); /*Add each workload (struct) to vector of workloads*/
                
        workloads_[workload].workloadID = workload;
        workloads_[workload].clusterID = 0;           /*Deliverbale is 1 cluster*/
        workloads_[workload].MinZ = 0;                
        workloads_[workload].MaxZ = tensorShape_[2]-1;  //output channels
        workloads_[workload].padTop = 0;              /*These are zero in PoC compiler - relevant after WW09*/
        workloads_[workload].padBottom = 0;           /*These are zero in PoC compiler - relevant after WW09*/
        workloads_[workload].padLeft = 0;             /*These are zero in PoC compiler - relevant after WW09*/
        workloads_[workload].padRight = 0; /*These are zero in PoC compiler - relevant after WW09*/
                
        workloads_[workload].MPEMode = mv::Matrix;        /*Matrix is MPE Mode (4,4)*/
                
       /* Converting the paritions returned by METIS 
        * into tensor coordinates and populating these fields of workload 
        */

        using xyz_type = decltype(mv::Workload::MinX);

        // NB: references (just shorter aliases for WL coordinates)
        xyz_type& wl_min_x = workloads_[workload].MinX;
        xyz_type& wl_min_y = workloads_[workload].MinY;
        xyz_type& wl_max_x = workloads_[workload].MaxX;
        xyz_type& wl_max_y = workloads_[workload].MaxY; // all zeros

        wl_min_x = std::numeric_limits<xyz_type>::max(); // set wl_min x to 32767
        wl_min_y = std::numeric_limits<xyz_type>::max();
        wl_max_x = -1;
        wl_max_y = -1;

        for (int i=0; i < metisGraph_->m_numberTensorVertices; i++) {

            if (metisGraph_->part[i] == workload) {

                int min_x = metisGraph_->node_coords[i].min_x();
                int max_x = metisGraph_->node_coords[i].max_x();
                int min_y = metisGraph_->node_coords[i].min_y();
                int max_y = metisGraph_->node_coords[i].max_y();
                                
                //points vector below in workloads stores all the points belonging to the workload.
                //Note: Each rectangle of mpeMode shape is stored with only 4 co-ordinates, the vertices
                //so, 16 elements are inside the mpeMode rectangle (4X4 or 16x1), represented by a rectangle of 4 vertices
                //Important note: the vertices correspond to the corner elements of the 16 element rectangle
                //so for a 4x4 rectangle, vertex co-ordinates could be (0,0) (0.3) (3,0) (3,3)
                //the rectangle is inclusive of the points at the vertices
                workloads_[workload].points.push_back(std::make_pair(min_x,min_y));
                workloads_[workload].points.push_back(std::make_pair(min_x,max_y-1));
                workloads_[workload].points.push_back(std::make_pair(max_x-1,min_y));
                workloads_[workload].points.push_back(std::make_pair(max_x-1,max_y-1));
                // NB: guard calling to std::min/max with parentheses,
                //     as they may mess with same-named macro on Windows
                wl_min_x = (std::min)(wl_min_x, static_cast<xyz_type>(min_x));
                wl_max_x = (std::max)(wl_max_x, static_cast<xyz_type>(max_x));
                wl_min_y = (std::min)(wl_min_y, static_cast<xyz_type>(min_y));
                wl_max_y = (std::max)(wl_max_y, static_cast<xyz_type>(max_y));
            }
        }

        /*At the edge of the x dimension*/
        if(wl_max_x == metisGraph_->tensorXDim) 
            wl_max_x = wl_max_x - 1;
        
        /*At the edge of the y dimension*/
        if(wl_max_y == metisGraph_->tensorYDim)
            wl_max_y = wl_max_y - 1;
        
        /*Now Need to detect if the workload border is in the middle of tensor, if so then subtract n_elem_x or n_elem_y */

        /*Workload border in the middle of the tensor therefore subtract 1 from x dimension*/
        if((wl_max_x < metisGraph_->tensorXDim) && (wl_max_x <  (metisGraph_->tensorXDim-1)))
            wl_max_x = wl_max_x - 1;
        
        /*Workload border in the middle of the tensor therefore subtract 1 from max_y and add 1 to min_y (think bottom left)*/
        if((wl_max_y < metisGraph_->tensorYDim) && (wl_max_y <  (metisGraph_->tensorYDim-1)) && (wl_max_y !=  (metisGraph_->tensorYDim-1)) && (wl_min_y !=  0)) { 
             wl_max_y = wl_max_y - 1;
             //wl_min_y = wl_min_y + 1;
        }

        /*Workload border in the middle of the tensor therefore subtract 1 from max_y and no need to change min_y as it is already 0 (think top left)*/
        if((wl_max_y < metisGraph_->tensorYDim) && (wl_max_y <  (metisGraph_->tensorYDim-1)) && (wl_max_y !=  (metisGraph_->tensorYDim-1)) && (wl_min_y ==  0)) { 
             wl_max_y = wl_max_y - 1;
        }
        // the workload vertices are nothing but min max values of the workload. if the workload is a rectangle
        //the minmax values are true vertices. If not, the vertices may not really exist in the workload
        // as the vertices are created using the MinX, MinY, MaxX, MaxY of all the coordinates of entire workload
        workloads_[workload].vertices.push_back(std::make_pair(wl_min_x, wl_min_y));
        workloads_[workload].vertices.push_back(std::make_pair(wl_min_x, wl_max_y));
        workloads_[workload].vertices.push_back(std::make_pair(wl_max_x, wl_min_y));
        workloads_[workload].vertices.push_back(std::make_pair(wl_max_x, wl_max_y));

        //add to the 'listOfworkloadLists' vector, the returned list of workloads from the polygonworkloadsplit function       
        listOfworkloadLists.push_back(mv::Workloads::polygonWorkloadSplit(pass, workloads_[workload], workloads_, mpeMode));

        pass.log(mv::Logger::MessageType::Debug, "\nworkload: " + std::to_string(workload));
        pass.log(mv::Logger::MessageType::Debug, " max_x: " + std::to_string(workloads_[workload].MaxX));
        pass.log(mv::Logger::MessageType::Debug, " min_x: " + std::to_string(workloads_[workload].MinX));
        pass.log(mv::Logger::MessageType::Debug, " max_y: " + std::to_string(workloads_[workload].MaxY));
        pass.log(mv::Logger::MessageType::Debug, " min_y: " + std::to_string(workloads_[workload].MinY));
        pass.log(mv::Logger::MessageType::Debug, " min_z: " + std::to_string(workloads_[workload].MinZ));
        pass.log(mv::Logger::MessageType::Debug, " max_z: " + std::to_string(workloads_[workload].MaxZ));
    }
    //clearing the workloads as they may have shaped like polygons (not rectangles)
    workloads_.clear();
    //adding the rectangle workloads into workloads_ list
    for (auto listIt = listOfworkloadLists.begin(); listIt != listOfworkloadLists.end(); listIt++)
    {
        for (auto it = listIt->begin(); it != listIt->end(); it++)
        {
            workloads_.push_back(*it);
        }
    }
}

std::vector<mv::Workload> mv::Workloads::polygonWorkloadSplit(const mv::pass::PassEntry &pass, mv::Workload &workload, std::vector<mv::Workload> &workloads_, std::pair<idx_t, idx_t> &mpeMode)
{

    /*------------------------------------------------------------------------------------------------------------
        1. check if the area of the rectangle and number of elements match, if matches, then the polygon is a rectangle
        2. If isn't equal, missing part of the rectangle could be at the vertex or/and along the edges.Note, current status is supporting only the 'points not in workload'
        being at the vertices. Need to support the case where the missing points may be on the perimeter but not the edge
        3. get the list of 'interesting points' - these points are the ones that are 1) closest to the missing vertices or/and 2) the inner points
        on the (cut) edge. Below is a possible metis partition with AD edge of the rectangle ABCD that has A1A2 cut out and A3D cut out. The interesting points are at the corners
        which are A1, A2, A3, and C1. Current implementation is only supporting missing point at vertex, so D (and A3,C1) but not A1,A2.
        4. For each interesting point, recursively partitions are made till each of the partition is a rectangle
        5. Select the parition scheme that gives minimum number of rectangles

        A* * * * * * *A1       A2* * * *A3      D
         *           *           *     *
         *           * * * * * * *     * * * * *C1
         *                                     *
        B* * * * * * * * * * * * * * * * * * * *C
        ------------------------------------------------------------------------------------------------------------
        */

    std::vector<mv::Workload> workloadFromAreaCheck, finalWorkloadList;
    std::vector<std::pair<int16_t, int16_t>> points_not_in_wl;
    //NB: Interesting points has a boolen value. This value indicates whether the point is at the end of a partition or at the beginning. If starting,
    // then the point has to be included in the partition
    std::vector<std::pair<std::pair<int16_t, int16_t>, bool>> interesting_points;
    // check if the area is equal to the number of points
    if (4 * workload.pointsTotal() == workload.area())
    {
        workloadFromAreaCheck.push_back(workload);
        return workloadFromAreaCheck;
    }
    else if (4 * workload.pointsTotal() != workload.area())
    {
        // find interesting points
        for (auto it = begin(workload.vertices); it != end(workload.vertices); ++it)
        {
            if (std::find(workload.points.begin(), workload.points.end(), std::make_pair(it->first, it->second)) == workload.points.end())
            {
                {
                    points_not_in_wl.push_back(*it);
                }
            }
        }
    }
    // interesting points calculation
    int16_t diff1, diff2;
    std::pair<int16_t, int16_t> save1, save2;
    bool intPoint1isAtstart, intPoint2isAtstart;
    for (auto it = points_not_in_wl.begin(); it != points_not_in_wl.end(); it++)
    {
        intPoint1isAtstart = false;
        intPoint2isAtstart = false;
        diff1 = INT16_MAX;
        diff2 = INT16_MAX;
        for (auto all_it = workload.points.begin(); all_it != workload.points.end(); all_it++)
        {
            // check the points that are on the perimeter. Check all the workload points that have same 'x'
            if (it->first == all_it->first)
            {
                // find the closest point (of the workload) to the 'points not in wl'
                if (diff1 > abs(it->second - all_it->second))
                {
                    diff1 = abs(it->second - all_it->second);
                    save1 = *all_it;
                    if (it->second < all_it->second)
                        intPoint1isAtstart = true;
                }
            }
            // check the points that are on the perimeter. Check all the workload points that have same 'y'
            else if (it->second == all_it->second)
            {
                // find the closest point (of the workload) to the 'points not in wl'
                if (diff2 > abs(it->first - all_it->first))
                {
                    diff2 = abs(it->first - all_it->first);
                    save2 = *all_it;
                    if (it->first < all_it->first)
                        intPoint2isAtstart = true;
                }
            }
        }
        interesting_points.push_back(std::make_pair(save1, intPoint1isAtstart));
        interesting_points.push_back(std::make_pair(save2, intPoint2isAtstart));
    }

    idx_t templistSize = INT16_MAX;
    for (auto int_it = interesting_points.begin(); int_it != interesting_points.end(); int_it++)
    {
        workloadFromAreaCheck = mv::Workloads::workloadSplitHelper(pass, workload, *int_it, mpeMode);
        if (templistSize > workloadFromAreaCheck.size())
        {
            finalWorkloadList = workloadFromAreaCheck;
            templistSize = workloadFromAreaCheck.size();
        }
    }
    return finalWorkloadList;
}

std::vector<mv::Workload> mv::Workloads::workloadSplitHelper(const mv::pass::PassEntry &pass, mv::Workload &workload, std::pair<std::pair<int16_t, int16_t>, bool> &interesting_point, std::pair<idx_t, idx_t> &mpeMode)
{
    mv::Workload workload_partition_1, workload_partition_2;
    workload_partition_1.points.clear();
    workload_partition_2.points.clear();
    // compared to POC, a change is made here to include the comparison of workload min and max values.
    // two possible scenarios: split along X or Y (== comparison along Y and X)
    // split along X is needed, if the interesting point is on the workload Ymin or Ymax
    // split along Y is needed, if the interesting point is on the workload Xmin or Xmax

    if (workload.MinX == interesting_point.first.first || workload.MaxX == interesting_point.first.first)
    {
        if (interesting_point.second)
        {
            for (auto it_all = workload.points.begin(); it_all != workload.points.end(); it_all++)
            {
                if (it_all->second >= (interesting_point.first.second))
                    workload_partition_1.points.push_back(*it_all);
                else
                    workload_partition_2.points.push_back(*it_all);
            }
        }
        else if (not interesting_point.second)
        {
            for (auto it_all = workload.points.begin(); it_all != workload.points.end(); it_all++)
            {
                if (it_all->second > (interesting_point.first.second))
                    workload_partition_1.points.push_back(*it_all);
                else
                    workload_partition_2.points.push_back(*it_all);
            }
        }
    }
    else if (workload.MaxY > 0)
    {
        if (interesting_point.second)
        {
            for (auto it_all = workload.points.begin(); it_all != workload.points.end(); it_all++)
            {
                if (it_all->first >= (interesting_point.first.first))
                    workload_partition_1.points.push_back(*it_all);
                else
                    workload_partition_2.points.push_back(*it_all);
            }
        }
        else if (not interesting_point.second)
        {
            for (auto it_all = workload.points.begin(); it_all != workload.points.end(); it_all++)
            {
                if (it_all->first > (interesting_point.first.first))
                    workload_partition_1.points.push_back(*it_all);
                else
                    workload_partition_2.points.push_back(*it_all);
            }
        }
    }

    workload_partition_1.setMinMaxAndVertices();
    workload_partition_2.setMinMaxAndVertices();
    std::vector<mv::Workload> final1;
    std::vector<mv::Workload> finalWorkloadList;
    // add what happens if the partitions are empty --- same as POC. This can be tricky as the size of hte vector in points total
    // may be the max number of elements allowed for that vector type
    // Add a 'throw exception' condition below
    if (workload_partition_1.pointsTotal() == 0 || workload_partition_2.pointsTotal() == 0)
        throw mv::RuntimeError(pass, "Inside workload splitter into rectangles. workload parition with non zero points can't exist");
    final1 = mv::Workloads::polygonWorkloadSplit(pass, workload_partition_2, workloads_, mpeMode);
    finalWorkloadList = mv::Workloads::polygonWorkloadSplit(pass, workload_partition_1, workloads_, mpeMode);
    finalWorkloadList.insert(finalWorkloadList.end(), final1.begin(), final1.end());

    return finalWorkloadList;
}

mv::CostFunctions mv::Workloads::getCostFunction(mv::Element& passDesc) const
{
    /*parse CostFunction from Comp Descriptor*/
    mv::CostFunctions costFunction = mv::CostFunctions::Balanced; //default
    if (passDesc.hasAttr("costfunction")) 
    {
        std::string sCostFunction = passDesc.get<std::string>("costfunction");
        if (sCostFunction == "balanced")
            costFunction = mv::CostFunctions::Balanced;
        else if (sCostFunction == "criticalpath")
            costFunction = mv::CostFunctions::CriticalPath;
        else if (sCostFunction == "minmax")
            costFunction = mv::CostFunctions::MinMaxWorkloads;
        else if (sCostFunction == "greedy")
            costFunction = mv::CostFunctions::Greedy;
        else 
            this->log(mv::Logger::MessageType::Warning, "Could not parse the Cost Function type (only \"balanced | criticalpath | minmax | greedy\" currently supported). Using \"Balanced\"...");
    }
    else 
        this->log(mv::Logger::MessageType::Info, "No Cost Function specified in descriptor, using \"Balanced\"...");
    return costFunction;
}

std::vector<std::string> mv::Workloads::getTensorSplitAlgorithms(mv::Element& passDesc) const
{
    /*parse TensorSplitAlgorithms from Compilation Descriptor*/
    std::vector<std::string> algorithms = {"Metis", "Rectangle", "Z-Tiling"}; //default
    if (passDesc.hasAttr("TensorSplitAlgorithms")) 
    {
        algorithms.clear();
        std::string sAlgorithms = passDesc.get<std::string>("TensorSplitAlgorithms");
        std::stringstream ss(sAlgorithms);
        while( ss.good() )
        {
            std::string tempStr;
            std::getline(ss, tempStr, ',');
            if (tempStr=="Metis" || tempStr=="Rectangle" || tempStr=="Z-Tiling")
                algorithms.push_back(tempStr);
            else
                this->log(mv::Logger::MessageType::Warning, "Could not parse the TensorSplitAlgorithms type (only \"Metis, Rectangle, Z-Tiling\" currently supported).");
        }
    }
    else 
        this->log(mv::Logger::MessageType::Info, "No TensorSplitAlgorithms specified in descriptor, using  \"Metis, Rectangle, Z-Tiling\"...");
    
    //if parsing problem, return all 3
    if (algorithms.size() == 0)
        algorithms = {"Metis", "Rectangle", "Z-Tiling"};
    return algorithms;
}

std::vector<float> mv::Workloads::getExecutionCycles() const
{
    return executionCycles_;
}

void mv::Workloads::setExecutionCycles(std::vector<float> val)
{
    executionCycles_ = val;
}

void mv::Workloads::generateExecutionCycles(std::vector<mv::Data::TensorIterator>& outputTensor, int nDPUxCluster, CostFunctions costFunction)
{
    // notes from POC compiler:  Execution time is bounded by
    //      sum(WL)/DPU <= T <= max(WL_max)*(P-1)/P
    if (nDPUxCluster < 1)
        throw mv::ArgumentError("Generate Workloads Pass", "nDPUxCluster", std::to_string(nDPUxCluster), "Invalid number of DPUs");

    std::vector<float> workloadsExecutionCycles;
    if (validateWorkloads(outputTensor))
    {   
        for(std::vector<mv::Workload>::iterator itWL = workloads_.begin(); itWL != workloads_.end(); ++itWL) 
        {
            std::pair <int,int> mpeMode (4, 4);
            if(itWL->MPEMode != mv::Matrix)
                mpeMode = {1,16};
            float height = itWL->MaxY - itWL->MinY; // + mpeMode.first;
            float width = itWL->MaxX - itWL->MinX; // + mpeMode.second;

            float sumExeCycles = ceil(outputTensor[0]->getShape()[2]/16.0) * ceil(height / mpeMode.first) * ceil(width / mpeMode.second);
            workloadsExecutionCycles.push_back(sumExeCycles);
        }
    }
    else
    {   //workload not schedulable
        workloadsExecutionCycles = {INFINITY};
    }
    
    float critical_wl = *std::max_element(workloadsExecutionCycles.begin(), workloadsExecutionCycles.end());
    //float lower_wl = *std::min_element(workloadsExecutionCycles.begin(), workloads_execution_cycles.end());

    float wl_sum = float(0);
    for (auto& cycles : workloadsExecutionCycles)
        wl_sum += cycles;

    float min_range = wl_sum/nDPUxCluster;
    float max_range = wl_sum/nDPUxCluster + critical_wl;

    if (costFunction == CostFunctions::Balanced)
    {
        float balancing = float(0.0);
        if (!std::isinf(wl_sum))
            balancing = wl_sum/(ceil(wl_sum/nDPUxCluster) * nDPUxCluster);

        executionCycles_ = {-balancing, -balancing};
    }
    else if(costFunction == CostFunctions::MinMaxWorkloads)
        executionCycles_ = {min_range, max_range};

    else if(costFunction == CostFunctions::CriticalPath)
    {
        if (nDPUxCluster == 1)
            executionCycles_ = {min_range, min_range};
        else
            executionCycles_ = {max_range, max_range};
    }

    else if(costFunction == CostFunctions::Greedy)
    {
        if (std::isinf(wl_sum))
            executionCycles_ = {INFINITY, INFINITY};
        else
        {
            float greedy = greedyTaskAssignment(nDPUxCluster, workloadsExecutionCycles);
            executionCycles_ = {greedy, greedy};
        }
    }
    else
        throw mv::ArgumentError("Generate Workloads Pass", "costFunction", "unknown", "Unsupported cost function");
}


/**
 * @brief
 * @param nProcessors is the number of computing resources
 * @param workloadCosts vector of workload costs
 */
float mv::Workloads::greedyTaskAssignment(int nProcessors, std::vector<float>& workloadCosts)
{
    std::priority_queue<int, std::vector<int>, std::greater<int> > exeCycles; //ascending sizes
    for (int i=0; i<nProcessors; ++i)
        exeCycles.push(0);
    
    for (size_t idxWorkload=0; idxWorkload<workloadCosts.size(); ++idxWorkload)
    {
        int smallestTime = exeCycles.top();
        exeCycles.pop();
        exeCycles.push(smallestTime + workloadCosts[idxWorkload]);
    }
    
    //return max value (ie, last value) in queue
    for (int i=0; i<nProcessors-1; ++i)
        exeCycles.pop();
    return exeCycles.top();
}


bool mv::Workloads::validateWorkloads(std::vector<mv::Data::TensorIterator>& inputTensor)
{
    return validateWorkloads(inputTensor[0]->getShape());
}

// Consider shapes equal if all dimensions equal but maybe 'N',
// e.g.: compare "NCWH" versus "CHW" by comparing 'C', 'H', 'W'
// Consider "CHW" and "HW" shapes equal is channels number = 1
// FIXME: need to know tensor orders to identify 'C', 'H', 'W'
//       (yet assume orders same: either "NCHW", "CHW" or "HW")
static bool equalShapes(const mv::Shape& a, const mv::Shape& b)
{
    auto m = (std::min)(a.ndims(), b.ndims());
    auto M = (std::max)(a.ndims(), b.ndims());

    if (m < 2 || m > 4 || M > 4)
        return false; // unsupported orders

    // ignore 4th dimension which must be 'N'
    for (unsigned i=0; i < m && i < 3; i++)
        if (a[i] != b[i])
            return false;

    auto& c = a.ndims() > b.ndims() ? a : b;

    // test channels (if compare CHW vs HW)
    for (unsigned i=m; i < M && i < 3; i++)
        if (c[i] != 1)
            return false;

    return true;
}

// Compute shape's volume without 'N' (batch) dimension
static size_t getTensorSize(const  mv::Shape & shape,
                            const std::string& order = "")
{
    // FIXME: analyze tensor's order to find which is 'N'
    //    (now assume order is "NCHW", or "CHW", or "HW")
    assert(order == "NCHW" || order == "CHW" || order == "HW" || order == "");

    size_t size = 1;
    for (unsigned i=0; i < shape.ndims() && i < 3; i++)
        size *= shape[i];

    return size;
}

bool mv::Workloads::validateWorkloads(const mv::Shape& shape)
{
    //    Check if the generated workloads are valid
    //    Check 1: the union of the workload have to make the whole tensor
    //          - Same Volume
    //          - Same vertices
    //    Check 2: no workload intersection

    // Check 0: empty workloads are not valid
    // Using size_t variable (nWorkloads) below, you may see a warning. Casting to double or int is unnecessary
    if (workloads_.size()  == 0)
    {
        this->log(mv::Logger::MessageType::Debug, "METIS partition failed because of total number of the partitions <=0");
        return false;
    }

    // Check 1: Volume of the tensor = sum of volumes of the individual workloads
    double vol = this->getAllWorkloadsVolume();
    std::size_t totalVol = getTensorSize(shape); // shape.totalSize() is wrong here as it counts 'N' (batch) dimension
    if (vol != totalVol)
    {
        this->log(mv::Logger::MessageType::Warning, "METIS partition failed because of volume differences. Original Tensor: " + 
                    std::to_string(shape.totalSize()) + " Partitioned Tensor: " + std::to_string(this->getAllWorkloadsVolume()));
        return false;
    }

    // Check for same vertices for each of the X, Y and X dimensions. This is done by comparing the shape of the inputTensor and min max of (all) workloads
    if (!equalShapes(this->getShapefromMinMax(), shape))
    {
        this->log(mv::Logger::MessageType::Warning, "METIS partition failed because vertices/bounds different between Original Tensor " + 
                                     shape.toString() + " and Partitioned Tensor " + this->getShapefromMinMax().toString());
        return false;
    }

    // Check 2: No intersection between workloads.
    // if (!this->noOverlap())
    // {
    //     this->log(mv::Logger::MessageType::Debug, "METIS partition failed because of overlap of paritions");
    //     return false;
    // }

    return true;
}

void mv::Workloads::addWorkload(mv::Workload workload)
{
    this->workloads_.push_back(workload);
}

//----------------------------------------------------------------------
//
//   Rectangle Heuristic
//
//----------------------------------------------------------------------


namespace mv {

    struct Shape2D
    {
        unsigned H; // height, aka X
        unsigned W; // width,      Y
    };

    using WorkloadContext = Shape2D;
    using WorkloadShape   = Shape2D;

    struct PaddingVariant
    {
        double          efficiency;
        WorkloadShape   original;
        WorkloadShape   reduced;
        WorkloadContext context;
    };


    // Elementary workload shapes
    using  WorkloadContextList = std::vector<WorkloadContext>;
    static WorkloadContextList context_list = {{4,  4}, {16, 1}}; // height x width


    static unsigned divRoundUp(unsigned x, unsigned m) { return (x + m - 1) / m; } // e.g. div(1, 2) = 0.5 -> 1
    static unsigned padRoundUp(unsigned x, unsigned m) { return divRoundUp(x, m) * m; } // pad(1, 2)       -> 2


    static double estimateEfficiency(const WorkloadShape& original,
                                     const WorkloadShape& padded)
    {
        double o_volume = original.H * original.W;
        double p_volume =   padded.H *   padded.W;
        return o_volume / p_volume;
    }


    static PaddingVariant selectPadding(const WorkloadShape      & original,
                                        const WorkloadContextList& context_list)
    {
        double best_efficiency = 0;
        PaddingVariant best_variant; // undefined

        for (auto context : context_list)
        {
            WorkloadShape padded;
            padded.H = padRoundUp(original.H, context.H);
            padded.W = padRoundUp(original.W, context.W);

            auto efficiency = estimateEfficiency(original, padded);

            if (best_efficiency < efficiency)
            {
                best_efficiency = efficiency;

                WorkloadShape reduced;
                reduced.H = padded.H / context.H;
                reduced.W = padded.W / context.W;
                best_variant = {efficiency, original, reduced, context};
            }
        }

        return best_variant;
    }

    static bool split_over_h = true;
    static bool split_over_w = true;

    static bool split_symmetric = false;

    using SplitFactors = std::pair<unsigned, unsigned>;
    using SplitFactorsList = std::vector<SplitFactors>;

    // enlist factors of N (who evenly divides value N)
    static SplitFactorsList getSplitFactors(unsigned N)
    {
        SplitFactorsList factors;
        unsigned i_max = std::ceil(std::sqrt(N));
        for (unsigned i=1; i <= i_max; i++)
        {
            if (N % i == 0)
            {
                unsigned j = N / i;
                SplitFactors f = std::make_pair(i, j);
                factors.push_back(f);
            }
        }
        return factors;
    }

    // lower estimate is better
    // w, h -- tensor shape
    // x, y -- split factors
    static double estimateSplitBalance(unsigned W, unsigned H, unsigned X, unsigned Y)
    {
        // FIXME: POC maps W, H to X, Y (I guess it must map W, H to Y, X)
        if (!split_over_h && Y > 1)
            return INFINITY;
        if (!split_over_w && X > 1)
            return INFINITY;
        if (H < Y || W < X)
            return INFINITY;
        return (W/X)*H + (H/Y)*W;
    }

    struct SplitVariant
    {
        SplitFactors factors;
        double cost_estimate;
    };

    static SplitVariant getBestSplitSymmetric(unsigned W, unsigned H, unsigned N)
    {
        SplitVariant best_variant;
        best_variant.cost_estimate = INFINITY;

        SplitFactorsList factors = getSplitFactors(N);
        for (auto f: factors)
        {
            auto X = std::get<0>(f);
            auto Y = std::get<1>(f);

            double cost0 = estimateSplitBalance(W, H, X, Y);
            if (best_variant.cost_estimate > cost0)
            {
                best_variant.cost_estimate = cost0;
                best_variant.factors = std::make_pair(X, Y);
            }

            double cost1 = estimateSplitBalance(W, H, Y, X);
            if (best_variant.cost_estimate > cost1)
            {
                best_variant.cost_estimate = cost1;
                best_variant.factors = std::make_pair(Y, X);
            }
        }

        return best_variant;
    }

    struct SplitSlice
    {
        unsigned x0, x1;
        unsigned y0, y1;
    };

    using SplitSliceList = std::vector<SplitSlice>;

    struct SplitSliceVariant
    {
        SplitSliceList slice_list;
        double      cost_estimate;
    };

    static SplitSliceVariant splitSliceSymmetric(unsigned W, unsigned H, unsigned N)
    {
        SplitVariant best_variant = getBestSplitSymmetric(W, H, N);
        double& cost_estimate = best_variant.cost_estimate;
        SplitFactors& factors = best_variant.factors;
        unsigned X = std::get<0>(factors);
        unsigned Y = std::get<1>(factors);

        SplitSliceVariant slice_list_variant;
        slice_list_variant.cost_estimate = cost_estimate;

        if (std::isinf(cost_estimate))
            return slice_list_variant; // empty slice list

        //FIXME: POC associates W, H with X, Y (I guss must associate with Y, X)

        unsigned dx = std::ceil(static_cast<double>(W) / X);
        unsigned dy = std::ceil(static_cast<double>(H) / Y);

        SplitSliceList slice_list; // empty
        for (unsigned x=0; x * dx < W; x++)
        for (unsigned y=0; y * dy < H; y++)
        {
            SplitSlice slice;
            slice.x0 = x * dx;
            slice.y0 = y * dy;
            slice.x1 = (std::min)((x + 1)*dx, W);
            slice.y1 = (std::min)((y + 1)*dy, H);
            slice_list.push_back(slice);
        }

        slice_list_variant.slice_list = slice_list;
        return slice_list_variant;
    }

    struct SplitVariantNonSymmetric : public SplitVariant
    {
        unsigned xss, yss;
        char mode;
    };

    static SplitVariantNonSymmetric getBestSplitNonSymmetric(unsigned W, unsigned H, unsigned N)
    {
        SplitVariantNonSymmetric best_variant;
        best_variant.cost_estimate = INFINITY; // worst

        SplitFactorsList factors = getSplitFactors(N - 1);
        for (auto f : factors)
        {
            auto K = std::get<0>(f);
            auto P = std::get<1>(f);
            if (K > P)
                std::swap(K, P); // ensure X <= Y

            if (K == 1)
                continue;

            unsigned a1 = std::ceil((std::max)(H, W) * static_cast<double>(K + 1) / N);
            unsigned a2 = std::ceil((std::min)(H, W) / static_cast<double>(K + 1));

            unsigned a3 = std::floor((std::min)(H, W) * static_cast<double>(K + 1) / N);
            unsigned a4 =  std::ceil((std::max)(H, W) / static_cast<double>(K + 1));

            if (H >= W)
            {
                double cost0 = estimateSplitBalance(    a3, H,   1, K+1)
                             + estimateSplitBalance(W - a3, H, P-1, K);
                if (best_variant.cost_estimate > cost0)
                {
                    best_variant.cost_estimate = cost0;
                    best_variant.factors = std::make_pair(P, K);
                    best_variant.xss = a3;
                    best_variant.yss = a4;
                    best_variant.mode = 'H';
                }

                double cost1 = estimateSplitBalance(W,     a1, K+1, 1)
                             + estimateSplitBalance(W, H - a1, K  , P-1);
                if (best_variant.cost_estimate > cost1)
                {
                    best_variant.cost_estimate = cost1;
                    best_variant.factors = std::make_pair(K, P);
                    best_variant.xss = a2;
                    best_variant.yss = a1;
                    best_variant.mode = 'W';
                }
            }
            else // if H < W
            {
                double cost2 = estimateSplitBalance(    a1, H,   1, K+1)
                             + estimateSplitBalance(W - a1, H, P-1, K);
                if (best_variant.cost_estimate > cost2)
                {
                    best_variant.cost_estimate = cost2;
                    best_variant.factors = std::make_pair(P, K);
                    best_variant.xss = a1;
                    best_variant.yss = a2;
                    best_variant.mode = 'H';
                }

                double cost3 = estimateSplitBalance(W,     a3, K+1, 1)
                             + estimateSplitBalance(W, H - a3, K  , P-1);
                if (best_variant.cost_estimate > cost3)
                {
                    best_variant.cost_estimate = cost3;
                    best_variant.factors = std::make_pair(K, P);
                    best_variant.xss = a4;
                    best_variant.yss = a3;
                    best_variant.mode = 'W';
                }
            }
        }

        return best_variant;
    }

    static SplitSliceVariant splitSliceNonSymmetric(unsigned W, unsigned H, unsigned N)
    {
        SplitVariantNonSymmetric best_split = getBestSplitNonSymmetric(W, H, N);
        double& cost_estimate = best_split.cost_estimate;
        SplitFactors& factors = best_split.factors;
        unsigned X = std::get<0>(factors);
        unsigned Y = std::get<1>(factors);
        unsigned xss = best_split.xss;
        unsigned yss = best_split.yss;
        char mode = best_split.mode;

        SplitSliceVariant slice_list_variant;
        slice_list_variant.cost_estimate = cost_estimate;

        if (std::isinf(cost_estimate))
            return slice_list_variant; // no slicing in fact

        //FIXME: POC associates W, H with X, Y (I guss must associate with Y, X)

        unsigned x_start = 0;
        unsigned y_start = 0;
        SplitSliceList slice_list; // empty

        if (mode == 'H')
        {
            for (unsigned y=0; y < Y+1; y++)
            {
                SplitSlice slice;
                slice.x0 = 0;
                slice.x1 = xss;
                slice.y0 = y * yss;
                slice.y1 = (std::min)((y + 1)*yss, H);
                slice_list.push_back(slice);
            }
            x_start = xss;
            Y -= 1;
        }
        else // if mode == 'W'
        {
            for (unsigned x=0; x < X+1; x++)
            {
                SplitSlice slice;
                slice.x0 = x * xss;
                slice.x1 = (std::min)((x + 1)*xss, W);
                slice.y0 = 0;
                slice.y1 = yss;
                slice_list.push_back(slice);
            }
            y_start = yss;
            X -= 1;
        }

        unsigned x_size = std::ceil(static_cast<double>(W - x_start) / X);
        unsigned y_size = std::ceil(static_cast<double>(H - y_start) / Y);

        for (unsigned x=0; x < X; x++)
            for (unsigned y=0; y < Y; y++)
            {
                SplitSlice slice;
                slice.x0 = x*x_size + x_start;
                slice.y0 = y*y_size + y_start;
                slice.x1 = (std::min)((x+1)*x_size + x_start, W);
                slice.y1 = (std::min)((y+1)*y_size + y_start, H);
                slice_list.push_back(slice);
            }

        slice_list_variant.slice_list = slice_list; // maybe std::move()
        return slice_list_variant;
    }

    using  WorkloadList = std::vector<Workload>;
    static WorkloadList generateWorkloadsFromSlices(const SplitSliceList& slice_list,
                                                    const PaddingVariant& padding,
                                                    unsigned Z=0)
    {
        WorkloadList workload_list;

        // FIXME: probably map W, H to Y, X (POC maps to X, Y)
        unsigned x_coef = std::ceil(static_cast<double>(padding.original.W) / padding.reduced.W);
        unsigned y_coef = std::ceil(static_cast<double>(padding.original.H) / padding.reduced.H);

        for (auto slice : slice_list)
        {
            unsigned x_min = slice.x0 * x_coef;
            unsigned y_min = slice.y0 * y_coef;
            unsigned x_max = slice.x1 * x_coef;
            unsigned y_max = slice.y1 * y_coef;

            Workload workload;

            workload.MinX = x_min;
            workload.MinY = y_min;
            workload.MaxX = (std::min)(x_max - 1, padding.original.W - 1);
            workload.MaxY = (std::min)(y_max - 1, padding.original.H - 1);

            // TODO: implement partitioning by Z (aka C=channels)
            workload.MinZ = 0;
            workload.MaxZ = Z ? Z - 1: 0;

            // FIXME: setup workload id
            // FIXME: adjust workloads padding
            workload_list.push_back(workload);
        }

        return workload_list;
    }

} // namespace mv


int mv::Workloads::partitionTensorWithRectangleHeuristic(idx_t nWorkloads, const mv::pass::PassEntry &pass)
{
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: layer=" + layerName_);

    // FIXME: need to know tensor order to find its dimensions: width, height,...
    // HACK: assume tensor order is "NCHW", so width=shape[0] and height=shape[1]
    unsigned C, H, W;
    if (tensorShape_.ndims() < 2) {
        pass.log(mv::Logger::MessageType::Error,
                 "RectangleHeuristic: too few tensor ndims=" + std::to_string(tensorShape_.ndims()));
        return METIS_ERROR;
    }
    W = tensorShape_[0];
    H = tensorShape_[1];
    C = tensorShape_.ndims() >= 3 ? tensorShape_[2] : 0;

    //
    // FIXME: POC compiler associates W, H with X, Y (I guess must be Y, X instead of X, Y)
    //
    WorkloadShape original_shape;
    original_shape.W = W; // width, aka X
    original_shape.H = H; // height,    Y
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: original_height=" + std::to_string(original_shape.H)
                                                             + ", original_width="  + std::to_string(original_shape.W));
    auto best_padding = selectPadding(original_shape, context_list);
    auto& reduced_shape = best_padding.reduced;
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: reduced_height=" + std::to_string(reduced_shape.H)
                                                             + ", reduced_width="  + std::to_string(reduced_shape.W));

    SplitSliceVariant slicing_variant = splitSliceSymmetric(reduced_shape.W, reduced_shape.H, nWorkloads);
    if (!split_symmetric)
    {
        SplitSliceVariant slicing_variant_2 = splitSliceNonSymmetric(reduced_shape.W, reduced_shape.H, nWorkloads);
        if (slicing_variant.cost_estimate > slicing_variant_2.cost_estimate)
            slicing_variant = slicing_variant_2;
    }
    if (std::isinf(slicing_variant.cost_estimate))
    {
        pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: cannot slice!");
        return METIS_ERROR;
    }
    SplitSliceList& slice_list = slicing_variant.slice_list;
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: slices=" + std::to_string(slice_list.size()));

    //
    // FIXME: see details inside code of generateWorkloadsFromSlices()
    //
    workloads_ = generateWorkloadsFromSlices(slice_list, best_padding, C);
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: done");

    return METIS_OK;
}
