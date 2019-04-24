#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include <metis.h>

#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

mv::Workloads::Workloads(const std::string& name, const mv::Shape& tensorShape, std::pair <int,int>& mpeMode):
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
        volume += (this->workloads_[i].MaxX - this->workloads_[i].MinX + 1) *
                  (this->workloads_[i].MaxY - this->workloads_[i].MinY + 1) *
                  (this->workloads_[i].MaxZ - this->workloads_[i].MinZ + 1);
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
void mv::Workloads::generateMetisGraph(void) const {

    /*Nodes in the graph*/
    std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph_->m_numberTensorVertices);
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
        if((*it%metisGraph_->m_xDim == 0) && (*it == 0)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + (metisGraph_->m_xDim); 
            adjncyIndex++;
        }
  
        /*Intermediate node left side*/ 
        if((*it%metisGraph_->m_xDim == 0) && ((*it + metisGraph_->m_xDim) < ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - metisGraph_->m_xDim;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + (metisGraph_->m_xDim); 
            adjncyIndex++;
        }

        /*Bottom left node*/
        if((*it%metisGraph_->m_xDim == 0) && ((*it + metisGraph_->m_xDim) > ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - metisGraph_->m_xDim;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
        }

        /*Top right node*/
        if(((*it - (metisGraph_->m_xDim-1)%metisGraph_->m_xDim == 0) && ((*it - (*it -(metisGraph_->m_xDim-1))) == metisGraph_->m_xDim -1) && ((*it-(metisGraph_->m_xDim-1) == 0)))) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + (metisGraph_->m_xDim); 
            adjncyIndex++;
        }

       /*Intermediate right side node*/
        if(((*it - (metisGraph_->m_xDim-1))%metisGraph_->m_xDim == 0) && ((*it - (*it -(metisGraph_->m_xDim-1))) == metisGraph_->m_xDim -1)  && ((*it-(metisGraph_->m_xDim-1) != 0))  && (*it %(nodeNumbers.size()-1) != 0)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - metisGraph_->m_xDim;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + (metisGraph_->m_xDim); 
            adjncyIndex++;
        }

        /*Bottm right node*/
        if(((*it - (metisGraph_->m_xDim-1))%metisGraph_->m_xDim == 0) && ((*it - (*it -(metisGraph_->m_xDim-1))) == metisGraph_->m_xDim -1) && (*it %(nodeNumbers.size()-1) == 0)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - (metisGraph_->m_xDim); 
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->xadj[xadjIndex] = adjncyIndex;
        }
        
        /*Middle nodes top row*/
        if(((*it)%metisGraph_->m_xDim != 0) && ((*it) < (metisGraph_->m_xDim - 1))) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + (metisGraph_->m_xDim); 
            adjncyIndex++;
        }

        /*Middle nodes bottom row*/
        if(((*it)%metisGraph_->m_xDim != 0) && ((*it) > ((int)nodeNumbers.size()-1) - metisGraph_->m_xDim) && ((*it) != ((int)nodeNumbers.size()-1))) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - (metisGraph_->m_xDim); 
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
        }

        /*Middle nodes not on bottom or top rows or the side columns*/
        if(((*it)%metisGraph_->m_xDim != 0) && ((*it) < ((int)nodeNumbers.size()-1) - metisGraph_->m_xDim) && ((*it) > (metisGraph_->m_xDim-1)) && ((*it+1)%metisGraph_->m_xDim != 0)) {

            metisGraph_->xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - (metisGraph_->m_xDim); 
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph_->adjncy[adjncyIndex] = *it + (metisGraph_->m_xDim); 
            adjncyIndex++;
        }
    }   
} 


int mv::Workloads::partitionTensorWithMETIS(idx_t nWorkloads, const mv::pass::PassEntry& pass) 
{
    METIS_SetDefaultOptions(metisGraph_->options);

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


void mv::Workloads::populateWorkloadsFromPartitions(idx_t nWorkloads, const mv::pass::PassEntry& pass) {
    
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

        for (int i=0; i < metisGraph_->m_numberTensorVertices; i++) {
            
            if (metisGraph_->part[i] == workload) {
                
                int min_x = metisGraph_->node_coords[i].min_x();
                int max_x = metisGraph_->node_coords[i].max_x();
                int min_y = metisGraph_->node_coords[i].min_y();
                int max_y = metisGraph_->node_coords[i].max_y();

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

mv::CostFunctions mv::Workloads::getCostFunction(mv::Element& passDesc, const mv::pass::PassEntry& pass) 
{
    /*parse CostFunction from Comp Descriptor*/
    CostFunctions costFunction = CostFunctions::Balanced; //default
    std::string sCostFunction = std::string(); 
    if (passDesc.hasAttr("costfunction")) 
    {
        sCostFunction = passDesc.get<std::string>("costfunction");
        if (sCostFunction == "balanced")
            costFunction = CostFunctions::Balanced;
        else if (sCostFunction == "criticalpath")
            costFunction = CostFunctions::CriticalPath;
        else if (sCostFunction == "minmax")
            costFunction = CostFunctions::MinMaxWorkloads;
        else if (sCostFunction == "greedy")
            costFunction = CostFunctions::Greedy;
        else 
            pass.log(mv::Logger::MessageType::Warning, "Could not parse the Cost Function type (only \"balanced | criticalpath | minmax | greedy\" currently supported). Using \"Balanced\"...");
    }
    else
        pass.log(mv::Logger::MessageType::Info, "No Cost Function specified in descriptor, using \"Balanced\"...");

}

std::vector<float> mv::Workloads::getExecutionCycles(std::vector<mv::Data::TensorIterator>& outputTensor, int nDPUxCluster, CostFunctions costFunction)
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
            if(itWL->MPEMode == mv::MPE_Mode::Matrix)
                mpeMode = {1,16};
            float height = itWL->MaxY - itWL->MinY + mpeMode.first;
            float width = itWL->MaxX - itWL->MinX + mpeMode.second;

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

        return {-balancing, -balancing};
    }
    else if(costFunction == CostFunctions::MinMaxWorkloads)
         return {min_range, max_range};

    else if(costFunction == CostFunctions::CriticalPath)
    {
        if (nDPUxCluster == 1)
            return {min_range, min_range};
        else
            return {max_range, max_range};
    }

    else if(costFunction == CostFunctions::Greedy)
    {
        if (std::isinf(wl_sum))
            return {INFINITY, INFINITY};
        else
        {
            float greedy = greedyTaskAssignment(nDPUxCluster, workloadsExecutionCycles);
            return {greedy, greedy};
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
    if (!this->noOverlap())
    {
        this->log(mv::Logger::MessageType::Debug, "METIS partition failed because of overlap of paritions");
        return false;
    }

    return true;
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
        for (unsigned x=0; x < X; x++)
        for (unsigned y=0; y < X; y++)
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
    SplitSliceList& slice_list = slicing_variant.slice_list;
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: slices=" + std::to_string(slice_list.size()));

    //
    // FIXME: see details inside code of generateWorkloadsFromSlices()
    //
    workloads_ = generateWorkloadsFromSlices(slice_list, best_padding, C);
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: done");

    return METIS_OK;
}
