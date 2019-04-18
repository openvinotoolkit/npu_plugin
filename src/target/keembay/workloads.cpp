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

mv::CostFunctions mv::Workloads::getCostFunction(mv::Element& passDesc, const mv::pass::PassEntry& pass) {

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

std::vector<float> mv::Workloads::getExecutionCycles(std::vector<mv::Data::TensorIterator>& outputTensor, mv::Workloads& workloads, int nDPUxCluster, std::pair <int,int> MPEMode, CostFunctions costFunction) {

    // notes from POC compiler:  Execution time is bounded by
    //      sum(WL)/DPU <= T <= max(WL_max)*(P-1)/P
    if (nDPUxCluster < 1)
        throw mv::ArgumentError("Generate Workloads Pass", "nDPUxCluster", std::to_string(nDPUxCluster), "Invalid number of DPUs");

    std::vector<float> workloadsExecutionCycles;
    if (validateWorkloads(outputTensor, workloads))
    {   
        for(std::vector<mv::Workload>::iterator itWL = workloads.getWorkloads().begin(); itWL != workloads.getWorkloads().end(); ++itWL) 
        {
            float height = itWL->MaxY - itWL->MinY + MPEMode.first;
            float width = itWL->MaxX - itWL->MinX + MPEMode.second;

            float sumExeCycles = ceil(outputTensor[0]->getShape()[2]/16.0) * ceil(height / MPEMode.first) * ceil(width / MPEMode.second);
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


bool  mv::Workloads::validateWorkloads(std::vector<mv::Data::TensorIterator>& inputTensor, mv::Workloads& workloads)
{
    //    Check if the generated workloads are valid
    //    Check 1: the union of the workload have to make the whole tensor
    //          - Same Volume
    //          - Same vertices
    //    Check 2: no workload intersection

    // Check 0: empty workloads are not valid
    // Using size_t variable (nWorkloads) below, you may see a warning. Casting to double or int is unnecessary
    if ((workloads.nWorkloads()) == 0)
    {
        workloads.log(mv::Logger::MessageType::Debug, "METIS partition failed because of total number of the partitions <=0");
        return false;
    }

    // Check 1: Volume of the tensor = sum of volumes of the individual workloads
    double vol = workloads.getAllWorkloadsVolume();
    std::size_t totalVol = inputTensor[0]->getShape().totalSize();
    if (inputTensor[0]->getShape().totalSize() != workloads.getAllWorkloadsVolume())
    {
        workloads.log(mv::Logger::MessageType::Warning, "METIS partition failed because of volume differences. Original Tensor: " + 
                    std::to_string(inputTensor[0]->getShape().totalSize()) + " Partitioned Tensor: " + std::to_string(workloads.getAllWorkloadsVolume()));
        return false;
    }

    // Check for same vertices for each of the X, Y and X dimensions. This is done by comparing the shape of the inputTensor and min max of (all) workloads
    if (workloads.getShapefromMinMax() != inputTensor[0]->getShape())
    {
        workloads.log(mv::Logger::MessageType::Warning, "METIS partition failed because vertices/bounds different between Original Tensor " + 
                                     inputTensor[0]->getShape().toString() + " and Partitioned Tensor " + workloads.getShapefromMinMax().toString());
        return false;
    }

    // Check 2: No intersection between workloads.
    if (!workloads.noOverlap())
    {
        workloads.log(mv::Logger::MessageType::Debug, "METIS partition failed because of overlap of paritions");
        return false;
    }

    return true;
}

           
           