#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include <metis.h>
mv::Workloads::Workloads(const std::string& name, const mv::Shape& tensorShape, std::pair <int,int>& mpeMode):
layerName_(name), tensorShape_(tensorShape), metisGraph_(new MetisGraphStructure(tensorShape, mpeMode))
{
    
}

mv::Workloads::~Workloads()
{
    std::cout << "calling class destructor" << std::endl;
    //delete metisGraph_;
}

mv::MetisGraphStructure& mv::Workloads::getMetisGraph()
{
    return *metisGraph_;
}

mv::Workload& mv::Workloads::operator[](int nworkload)
{

    return const_cast<Workload&>(static_cast<const Workloads*>(this)->operator[](nworkload));

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

std::vector<mv::Workload>& mv::Workloads::getWorkloads()
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
    for (std::size_t i = 0; i < this->nWorkloads(); ++i) {
        volume += (this->workloads_[i].MaxX - this->workloads_[i].MinX + 1) * (this->workloads_[i].MaxY - this->workloads_[i].MinY + 1) * (this->workloads_[i].MaxZ - this->workloads_[i].MinZ + 1);
    }
    return volume;
}

bool mv::Workloads::noOverlap() const
{
    bool noIntersect = false;
    for(std::size_t i=0; i< this->nWorkloads(); i++)
        {
            for(std::size_t j=0;j<this->nWorkloads();j++){
                if(i==j)
                {
                    continue;
                }
        // applying De Morgan's law ((A U B)' == A' ): Two rectangles donot overlap if one rectangle's minimum in a dimension is greater than the other rectangle's maximum in that dimension
        // check to be done for both the X and Y dimension.
                noIntersect = noIntersect || this->workloads_[i].MinX > this->workloads_[j].MaxX ||
                        this->workloads_[j].MinX > this->workloads_[i].MaxX ||
                        this->workloads_[i].MinY > this->workloads_[j].MaxY ||
                        this->workloads_[j].MinY > this->workloads_[i].MaxY ;
            
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
    minMax.push_back(maxX - minX + 1);    
    minMax.push_back(maxY - minY + 1);    
    minMax.push_back(maxZ - minZ + 1);   
    return mv::Shape(minMax);
}

/**
 * @brief Creates a METIS adjacency structure of a graph as per 23/45 METIS manual. 
 * @brief Representing the lattic structure of the tensor shape (in the X-Y corrdinate) 
 * @param metisGraph - a struct containing necessary parameters to pass to METIS
 * @return None
 * 
 */
void mv::Workloads::generateMetisGraph(MetisGraphStructure& metisGraph) {

    /*Nodes in the graph*/
    std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph.m_numberTensorVertices);
    int adjncyIndex = 0;
    int xadjIndex = 0;
    
    /* A Sample Graph
     * 0---1---2---3---4
     * |   |   |   |   |
     * 5---6---7---8---9
     * |   |   |   |   |
     * 10--11---12-13--14
     */

    for (auto it = nodeNumbers.begin(); it != nodeNumbers.end(); it++) {

        /*Top left node*/ 
        if((*it%metisGraph.m_xDim == 0) && (*it == 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }
  
        /*Intermediate node left side*/ 
        if((*it%metisGraph.m_xDim == 0) && ((*it + metisGraph.m_xDim) < ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - metisGraph.m_xDim;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

        /*Bottom left node*/
        if((*it%metisGraph.m_xDim == 0) && ((*it + metisGraph.m_xDim) > ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - metisGraph.m_xDim;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
        }

        /*Top right node*/
        if(((*it - (metisGraph.m_xDim-1)%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1) && ((*it-(metisGraph.m_xDim-1) == 0)))) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

       /*Intermediate right side node*/
        if(((*it - (metisGraph.m_xDim-1))%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1)  && ((*it-(metisGraph.m_xDim-1) != 0))  && (*it %(nodeNumbers.size()-1) != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - metisGraph.m_xDim;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

        /*Bottm right node*/
        if(((*it - (metisGraph.m_xDim-1))%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1) && (*it %(nodeNumbers.size()-1) == 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - (metisGraph.m_xDim); 
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.xadj[xadjIndex] = adjncyIndex;
        }
        
        /*Middle nodes top row*/
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) < (metisGraph.m_xDim - 1))) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

        /*Middle nodes bottom row*/
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) > ((int)nodeNumbers.size()-1) - metisGraph.m_xDim) && ((*it) != ((int)nodeNumbers.size()-1))) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - (metisGraph.m_xDim); 
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
        }

        /*Middle nodes not on bottom or top rows or the side columns*/
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) < ((int)nodeNumbers.size()-1) - metisGraph.m_xDim) && ((*it) > (metisGraph.m_xDim-1)) && ((*it+1)%metisGraph.m_xDim != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - (metisGraph.m_xDim); 
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }
    }   
} 


int mv::Workloads::partitionTensorMETIS(MetisGraphStructure& metisGraph, idx_t nWorkloads) 
{
    METIS_SetDefaultOptions(metisGraph.options);

    /*METIS call*/
    int res = METIS_PartGraphRecursive(&metisGraph.m_numberTensorVertices,&metisGraph.nWeights, metisGraph.xadj, metisGraph.adjncy,
                    metisGraph.vwgt, NULL, NULL, &nWorkloads, NULL,
				    NULL, metisGraph.options, &metisGraph.objval, metisGraph.part);
                    
    return res;
}

/*TODO update*/
idx_t mv::Workloads::getNWorkloads(const mv::Shape& tensorShape, int nDPUxCluster) {
    
    return round(nDPUxCluster/2)*2; 
}


void mv::Workloads::populateWorkloadsFromPartitions(MetisGraphStructure& metisGraph, idx_t nWorkloads, const mv::pass::PassEntry& pass) {
    
    /*In some cases METIS might return a number or partitions (workloads) less than you specified (i.e. small tensor and large number of partitions*/
    /*This needs to be handled here for now assuming number of partitions is the number or workloads*/
            
    for(int workload = 0; workload < nWorkloads; workload++) { 
        
        getWorkloads().push_back(mv::Workload()); /*Add each workload (struct) to vector of workloads*/
                
        getWorkloads()[workload].workloadID = workload;
        getWorkloads()[workload].clusterID = 0;           /*WW09 deliverbale is 1 cluster*/
        getWorkloads()[workload].MinZ = 0;                /*WW09 deliverbale is less than 16 channels*/
        getWorkloads()[workload].MaxZ = tensorShape_[2] -1;  //output channels
        getWorkloads()[workload].padTop = 0;              /*These are zero in PoC compiler - relevant after WW09*/
        getWorkloads()[workload].padBottom = 0;           /*These are zero in PoC compiler - relevant after WW09*/
        getWorkloads()[workload].padLeft = 0;             /*These are zero in PoC compiler - relevant after WW09*/
        getWorkloads()[workload].padRight = 0;            /*These are zero in PoC compiler - relevant after WW09*/
                
        getWorkloads()[workload].MPEMode = mv::Matrix;        /*Matrix is MPE Mode (4,4)*/
                
       /* Converting the paritions returned by METIS 
        * into tensor coordinates and populating these fields of workload 
        */

        using xyz_type = decltype(mv::Workload::MinX);

        // NB: references (just shorter aliases for WL coordinates)
        xyz_type& wl_min_x = getWorkloads()[workload].MinX;
        xyz_type& wl_min_y = getWorkloads()[workload].MinY;
        xyz_type& wl_max_x = getWorkloads()[workload].MaxX;
        xyz_type& wl_max_y = getWorkloads()[workload].MaxY;

        wl_min_x = std::numeric_limits<xyz_type>::max();
        wl_min_y = std::numeric_limits<xyz_type>::max();
        wl_max_x = -1;
        wl_max_y = -1;

        for (int i=0; i < metisGraph.m_numberTensorVertices; i++) {
            
            if (metisGraph.part[i] == workload) {
                
                int min_x = metisGraph.node_coords[i].min_x();
                int max_x = metisGraph.node_coords[i].max_x();
                int min_y = metisGraph.node_coords[i].min_y();
                int max_y = metisGraph.node_coords[i].max_y();

                // NB: guard calling to std::min/max with parentheses,
                //     as they may mess with same-named macro on Windows
                wl_min_x = (std::min)(wl_min_x, static_cast<xyz_type>(min_x));
                wl_max_x = (std::max)(wl_max_x, static_cast<xyz_type>(max_x));
                wl_min_y = (std::min)(wl_min_y, static_cast<xyz_type>(min_y));
                wl_max_y = (std::max)(wl_max_y, static_cast<xyz_type>(max_y));
            }
        }
        pass.log(mv::Logger::MessageType::Debug, "\nworkload: " + std::to_string(workload));
        pass.log(mv::Logger::MessageType::Debug, " min_x: " + std::to_string(getWorkloads()[workload].MinX));
        pass.log(mv::Logger::MessageType::Debug, " max_x: " + std::to_string(getWorkloads()[workload].MaxX));
        pass.log(mv::Logger::MessageType::Debug, " min_y: " + std::to_string(getWorkloads()[workload].MinY));
        pass.log(mv::Logger::MessageType::Debug, " max_y: " + std::to_string(getWorkloads()[workload].MaxY));
        pass.log(mv::Logger::MessageType::Debug, " min_z: " + std::to_string(getWorkloads()[workload].MinZ));
        pass.log(mv::Logger::MessageType::Debug, " max_z: " + std::to_string(getWorkloads()[workload].MaxZ));
    }
}