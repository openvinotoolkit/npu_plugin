#include "include/mcm/target/kmb/workloads.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include <algorithm>
#include <sstream>

mv::Workloads::~Workloads()
{
}

mv::Workloads::Workloads(const std::string& name, const mv::Shape& tensorShape):
    layerName_(name), tensorShape_(tensorShape)
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
    return std::to_string(this->nWorkloads());
}

std::string mv::Workloads::toLongString() const
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
        output += "MPEMode " + std::to_string(this->workloads_[i].MPEMode) + ", ";
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
        auto volX = this->workloads_[i].MaxX - this->workloads_[i].MinX + 1;
        auto volY = this->workloads_[i].MaxY - this->workloads_[i].MinY + 1;
        auto volZ = this->workloads_[i].MaxZ - this->workloads_[i].MinZ + 1;
        volume += volX * volY * volZ;
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
 * @brief Generate a pool of possible workload number for a tensor
 * @param A tensor
 * @param Maximum number or workloads
 * @return A pool of possible splits (possible workloads)
 */
const std::vector<int> mv::Workloads::getWorkloadSplitPool(const Tensor& tensor, int nDPUxCluster, mv::DPUModeList dpuModeList, int maxSplits)
{
    std::vector<int> splitPool = {1};
    std::vector<int> maxSplitsXY;

    /*maxSplitsXY*/
    double xDim = tensor.getShape()[0];
    double yDim = tensor.getShape()[1];

    for(std::size_t i = 0; i < dpuModeList.size(); i++)
        maxSplitsXY.push_back(ceil((yDim/dpuModeList[i].H)  * ceil(xDim/dpuModeList[i].W)));

    /*maxSplitsZ*/
    int maxSplitsZ = ceil(tensor.getShape()[2]/16);

    /*Z splits*/
    if((maxSplitsZ < maxSplits) && (maxSplitsZ >1))
    {
        splitPool.push_back(maxSplitsZ);
        do
        {
            maxSplitsZ = maxSplitsZ >> 1;
            splitPool.push_back(maxSplitsZ);
        }
        while ((maxSplitsZ >> 1) > 1);
    }

    /*DpuMul splits*/
    for(int i = nDPUxCluster; i <= (maxSplits - nDPUxCluster) ; i+=nDPUxCluster)
        splitPool.push_back(i);

    /*XY splits*/
    for(std::size_t i = 0; i < dpuModeList.size(); i++) {
        for(int j = 0; j < (int)ceil(log2(maxSplitsXY[i])); j++)
            if(((maxSplitsXY[i]%(int)std::pow(2,j)) == 0) && (maxSplitsXY[i]/(std::pow(2,j)) <= maxSplits))
            splitPool.push_back(maxSplitsXY[i]/std::pow(2,j));
    }

    /*sort*/
    sort(splitPool.begin(), splitPool.end());

    /*Erase duplicates*/
    splitPool.erase(std::unique( splitPool.begin(), splitPool.end() ), splitPool.end());

    /*If the split pool is empty then make the default number of workloads be 4*/
    if(splitPool.empty())
        splitPool.push_back(4);

    return splitPool;
}

std::vector<mv::Workload> mv::Workloads::polygonWorkloadSplit(const mv::pass::PassEntry &pass, mv::Workload &workload, std::vector<mv::Workload> &workloads_, mv::DPUMode &mpe_mode)
{
    /*------------------------------------------------------------------------------------------------------------
        1.Missing part of the rectangle could be at the vertex or/and along the edges.Note, current status is supporting only the 'points not in workload'
        being at the vertices. Need to support the case where the missing points may be on the perimeter but not the edge
        2. If there are no points in the 'points not in the workload', it means the workload is already a rectangle.
        3. get the list of 'interesting points' - these points are the ones that are 1) closest to the missing vertices or/and 2) the inner points
        on the (cut) edge. Below is a possible partition with AD edge of the rectangle ABCD that has A1A2 cut out and A3D cut out. The interesting points are at the corners
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
    // find if any vertices are missing
    for (auto it = begin(workload.vertices); it != end(workload.vertices); ++it)
    {
        if (std::find(workload.points.begin(), workload.points.end(), std::make_pair(it->first, it->second)) == workload.points.end())
        {
            {
                points_not_in_wl.push_back(*it);
            }
        }
    }
    if (points_not_in_wl.empty())
    {
        workloadFromAreaCheck.push_back(workload);
        return workloadFromAreaCheck;

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
                    //            // point (no matter if it is closer to Ymin or Ymax), to have boolean as 'true' so that the point always is included in the right partition
                    if ((it->second < all_it->second))
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

    size_t templistSize = INT16_MAX;
    for (auto int_it = interesting_points.begin(); int_it != interesting_points.end(); int_it++)
    {
        workloadFromAreaCheck = mv::Workloads::workloadSplitHelper(pass, workload, *int_it, mpe_mode);
        if (templistSize > workloadFromAreaCheck.size())
        {
            finalWorkloadList = workloadFromAreaCheck;
            templistSize = workloadFromAreaCheck.size();
        }
    }

    return finalWorkloadList;
}

std::vector<mv::Workload> mv::Workloads::workloadSplitHelper(const mv::pass::PassEntry &pass, mv::Workload &workload, std::pair<std::pair<int16_t, int16_t>, bool> &interesting_point, mv::DPUMode &mpe_mode)
{
    mv::Workload workload_partition_1, workload_partition_2;
    workload_partition_1.points.clear();
    workload_partition_2.points.clear();
    // compared to POC, a change is made here to include the comparison of workload min and max values.
    // two possible scenarios: split along X or Y (== comparison along Y and X)
    // split along X is needed, if the interesting point is on the workload Ymin or Ymax
    // split along Y is needed, if the interesting point is on the workload Xmin or Xmax

    if ((workload.MinX == interesting_point.first.first) || (workload.MaxX == interesting_point.first.first))
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
    else if (workload.MaxY >= 0)
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
    if (workload_partition_1.pointsTotal() == 0 || workload_partition_2.pointsTotal() == 0) {
        pass.log(mv::Logger::MessageType::Warning, "Inside workload splitter into rectangles. workload parition with zero points can't exist, this should produce invalid workloads");
        return finalWorkloadList;
    }

    final1 = mv::Workloads::polygonWorkloadSplit(pass, workload_partition_2, workloads_, mpe_mode);
    finalWorkloadList = mv::Workloads::polygonWorkloadSplit(pass, workload_partition_1, workloads_, mpe_mode);
    finalWorkloadList.insert(finalWorkloadList.end(), final1.begin(), final1.end());

    return finalWorkloadList;
}

  /**
    * @brief Returns the cost function to use for execution cycles
    */
mv::CostFunctions mv::Workloads::getCostFunction(mv::Element& passDesc, const mv::pass::PassEntry& pass)
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
            pass.log(mv::Logger::MessageType::Warning, "Could not parse the Cost Function type (only \"balanced | criticalpath | minmax | greedy\" currently supported). Using \"Balanced\"...");
    }
    else
        pass.log(mv::Logger::MessageType::Info, "No Cost Function specified in descriptor, using \"Balanced\"...");
    return costFunction;
}

std::vector<float> mv::Workloads::getExecutionCycles() const
{
    return executionCycles_;
}

float mv::Workloads::getMeanExecutionCycles() const
{
    return meanExecutionCycles_;
}

void mv::Workloads::setExecutionCycles(std::vector<float> val)
{
    executionCycles_ = val;
}

void mv::Workloads::generateExecutionCycles(std::vector<mv::Workloads>& workloadsVector, int nDPUxCluster, CostFunctions costFunction, float pixelCost, int workloadCost)
{
    /* Execution time is bounded by
     * sum(WL)/DPU <= T <= max(WL_max)*(P-1)/P
    */

    /*Individual workload execution cycles*/
    std::vector<float> workloadsExecutionCycles;
    float min_range = 0;
    float max_range = 0;
    float sumExeCycles = 0;
    float workload_sum_  = 0;

    if (nDPUxCluster < 1)
        throw mv::ArgumentError("Generate Workloads Pass", "nDPUxCluster", std::to_string(nDPUxCluster), "Invalid number of DPUs");

    /*For each workload instance*/
    for(auto itWorkloads = workloadsVector.begin(); itWorkloads != workloadsVector.end(); itWorkloads++) {

        workloadsExecutionCycles.clear();
        min_range = 0;
        max_range = 0;
        sumExeCycles = 0;
        workload_sum_  = 0;

        /*Calculate the cost for each of the individual workloads (rectangles) */
        for(auto itworkload = itWorkloads->workloads_.begin(); itworkload != itWorkloads->workloads_.end(); ++itworkload) {
            std::pair <int,int> mpeMode (4, 4);

            if(itworkload->MPEMode == mv::Vector)
                mpeMode = {1,16};
            else if (itworkload->MPEMode == mv::Matrix)
                mpeMode = {4,4};
            else
                mpeMode = {1,4};


            float height = (itworkload->MaxY+1) - itworkload->MinY; // + mpeMode.first;
            float width = (itworkload->MaxX+1) - itworkload->MinX; // + mpeMode.second;

            sumExeCycles = ceil((itworkload->MaxZ-itworkload->MinZ)/16.0) * ceil(height / mpeMode.first) * ceil(width / mpeMode.second) * pixelCost;
            sumExeCycles += workloadCost;
            workloadsExecutionCycles.push_back(sumExeCycles);
        }

        /*Critical workload*/
        itWorkloads->critical_workload_ = *std::max_element(workloadsExecutionCycles.begin(), workloadsExecutionCycles.end());

        /*Workload sum*/
        for (auto& cycles : workloadsExecutionCycles)
            itWorkloads->workload_sum_ += cycles;

        min_range = itWorkloads->workload_sum_/nDPUxCluster;
        max_range = itWorkloads->workload_sum_/nDPUxCluster + itWorkloads->critical_workload_;

        /*Cost function*/
        if(costFunction == CostFunctions::CriticalPath) {

        /*Store the excutions cycles in the object*/
        if (nDPUxCluster == 1)
            itWorkloads->executionCycles_ = {min_range, min_range};
        else
            itWorkloads->executionCycles_ = {max_range, max_range};
        }

        else if (costFunction == CostFunctions::Balanced)
        {
            float balancing = float(0.0);
            if (!std::isinf(itWorkloads->workload_sum_))
                balancing = itWorkloads->workload_sum_/(ceil(itWorkloads->workload_sum_/nDPUxCluster) * nDPUxCluster);

            itWorkloads->executionCycles_ = {-balancing, -balancing};
        }

        else if (costFunction == CostFunctions::MinMaxWorkloads)
            itWorkloads->executionCycles_ = {min_range, max_range};

        else if (costFunction == CostFunctions::Greedy)
        {
            if (std::isinf(itWorkloads->workload_sum_))
                itWorkloads->executionCycles_ = {INFINITY, INFINITY};
            else
            {
                float greedy = greedyTaskAssignment(nDPUxCluster, workloadsExecutionCycles);
                itWorkloads->executionCycles_ = {greedy, greedy};
            }
        }
        else
            throw mv::ArgumentError("Generate Workloads Pass", "costFunction", "unknown", "Unsupported cost function");

        /*Calculate the mean of execution cycles*/
        itWorkloads->meanExecutionCycles_ = std::accumulate(itWorkloads->executionCycles_.begin(), itWorkloads->executionCycles_.end(), 0.0) / itWorkloads->executionCycles_.size();
    }
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
        this->log(mv::Logger::MessageType::Debug, "Partition failed because of total number of the partitions <=0");
        return false;
    }

    // Check 1: Volume of the tensor = sum of volumes of the individual workloads
    double vol = this->getAllWorkloadsVolume();
    std::size_t totalVol = getTensorSize(shape); // shape.totalSize() is wrong here as it counts 'N' (batch) dimension
    if (vol != totalVol)
    {
        this->log(mv::Logger::MessageType::Warning, "Partition failed because of volume differences. Original Tensor: " +
                    std::to_string(shape.totalSize()) + " Partitioned Tensor: " + std::to_string(this->getAllWorkloadsVolume()));
        return false;
    }

    // Check for same vertices for each of the X, Y and X dimensions. This is done by comparing the shape of the inputTensor and min max of (all) workloads
    if (!equalShapes(this->getShapefromMinMax(), shape))
    {
        this->log(mv::Logger::MessageType::Warning, "Partition failed because vertices/bounds different between Original Tensor " +
                                     shape.toString() + " and Partitioned Tensor " + this->getShapefromMinMax().toString());
        return false;
    }

    // Check 2: No intersection between workloads.
    if (!this->noOverlap())
    {
        this->log(mv::Logger::MessageType::Debug, "Partition failed because of overlap of paritions");
        return false;
    }

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
        double         efficiency;
        WorkloadShape  original;
        WorkloadShape  padded;
        WorkloadShape  reduced;
        mv::DPUMode    mode;
    };


    static unsigned divRoundUp(unsigned x, unsigned m) { return (x + m - 1) / m; } // e.g. div(1, 2) = 0.5 -> 1
    static unsigned padRoundUp(unsigned x, unsigned m) { return divRoundUp(x, m) * m; } // pad(1, 2)       -> 2


    static double estimateEfficiency(const WorkloadShape& original,
                                     const WorkloadShape& padded)
    {
        double o_volume = original.H * original.W;
        double p_volume =   padded.H *   padded.W;
        return o_volume / p_volume;
    }


    static PaddingVariant selectPadding(const WorkloadShape  & original,
                                        const mv::DPUModeList& mode_list)
    {
        double best_efficiency = 0;
        PaddingVariant best_variant; // undefined

        for (auto mode : mode_list)
        {
            WorkloadShape padded;
            padded.H = padRoundUp(original.H, mode.H);
            padded.W = padRoundUp(original.W, mode.W);

            auto efficiency = estimateEfficiency(original, padded);

            if (best_efficiency < efficiency)
            {
                best_efficiency = efficiency;

                WorkloadShape reduced;
                reduced.H = padded.H / mode.H;
                reduced.W = padded.W / mode.W;
                best_variant = {efficiency, original, padded, reduced, mode};
            }
        }

        return best_variant;
    }


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
    static double estimateSplitBalance(unsigned W, unsigned H, unsigned X, unsigned Y,
                                       bool split_over_h, bool split_over_w)
    {
        // FIXME: POC maps W, H to X, Y (I guess it must map W, H to Y, X)
        if (!split_over_h && Y > 1)
            return INFINITY;
        if (!split_over_w && X > 1)
            return INFINITY;
        if (H < Y || W < X)
            return INFINITY;
        return (W%X)*H + (H%Y)*W;
    }

    struct SplitVariant
    {
        SplitFactors factors;
        double cost_estimate;
    };

    static SplitVariant getBestSplitSymmetric(unsigned W, unsigned H, unsigned N,
                                              bool split_over_h, bool split_over_w)
    {
        SplitVariant best_variant;
        best_variant.cost_estimate = INFINITY;

        SplitFactorsList factors = getSplitFactors(N);
        for (auto f: factors)
        {
            auto X = std::get<0>(f);
            auto Y = std::get<1>(f);

            double cost0 = estimateSplitBalance(W, H, X, Y, split_over_h, split_over_w);
            if (best_variant.cost_estimate > cost0)
            {
                best_variant.cost_estimate = cost0;
                best_variant.factors = std::make_pair(X, Y);
            }

            double cost1 = estimateSplitBalance(W, H, Y, X, split_over_h, split_over_w);
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

    static SplitSliceVariant splitSliceSymmetric(unsigned W, unsigned H, unsigned N,
                                                 bool split_over_h, bool split_over_w)
    {
        SplitVariant best_variant = getBestSplitSymmetric(W, H, N, split_over_h, split_over_w);
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

    static SplitVariantNonSymmetric getBestSplitNonSymmetric(unsigned W, unsigned H, unsigned N,
                                                             bool split_over_h, bool split_over_w)
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
                double cost0 = estimateSplitBalance(    a3, H,   1, K+1, split_over_h, split_over_w)
                             + estimateSplitBalance(W - a3, H, P-1, K  , split_over_h, split_over_w);
                if (best_variant.cost_estimate > cost0)
                {
                    best_variant.cost_estimate = cost0;
                    best_variant.factors = std::make_pair(P, K);
                    best_variant.xss = a3;
                    best_variant.yss = a4;
                    best_variant.mode = 'H';
                }

                double cost1 = estimateSplitBalance(W,     a1, K+1, 1  , split_over_h, split_over_w)
                             + estimateSplitBalance(W, H - a1, K  , P-1, split_over_h, split_over_w);
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
                double cost2 = estimateSplitBalance(    a1, H,   1, K+1, split_over_h, split_over_w)
                             + estimateSplitBalance(W - a1, H, P-1, K  , split_over_h, split_over_w);
                if (best_variant.cost_estimate > cost2)
                {
                    best_variant.cost_estimate = cost2;
                    best_variant.factors = std::make_pair(P, K);
                    best_variant.xss = a1;
                    best_variant.yss = a2;
                    best_variant.mode = 'H';
                }

                double cost3 = estimateSplitBalance(W,     a3, K+1, 1  , split_over_h, split_over_w)
                             + estimateSplitBalance(W, H - a3, K  , P-1, split_over_h, split_over_w);
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

    static SplitSliceVariant splitSliceNonSymmetric(unsigned W, unsigned H, unsigned N,
                                                    bool split_over_h, bool split_over_w)
    {
        SplitVariantNonSymmetric best_split = getBestSplitNonSymmetric(W, H, N,
                                                    split_over_h, split_over_w);
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
            for (unsigned y=0; y * yss < H; y++)
            {
                SplitSlice slice;
                slice.x0 = 0;
                slice.x1 = xss;
                slice.y0 = y * yss;
                slice.y1 = (std::min)((y + 1)*yss, H);
                slice_list.push_back(slice);
            }
            x_start = xss;
            X -= 1;
        }
        else // if mode == 'W'
        {
            for (unsigned x=0; x * xss < W; x++)
            {
                SplitSlice slice;
                slice.x0 = x * xss;
                slice.x1 = (std::min)((x + 1)*xss, W);
                slice.y0 = 0;
                slice.y1 = yss;
                slice_list.push_back(slice);
            }
            y_start = yss;
            Y -= 1;
        }

        unsigned x_size = std::ceil(static_cast<double>(W - x_start) / X);
        unsigned y_size = std::ceil(static_cast<double>(H - y_start) / Y);

        for (unsigned x=0; x*x_size + x_start < W; x++)
            for (unsigned y=0; y*y_size + y_start < H; y++)
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
    static WorkloadList generateWorkloadsFromSlices(const mv::DPUModeList& mode_list, const SplitSliceList& slice_list,
                                                    const PaddingVariant& padding, const size_t nWorkloads,
                                                    unsigned Z=0)
    {
        WorkloadList workload_list;

        // FIXME: probably map W, H to Y, X (POC maps to X, Y)
        unsigned x_coef = std::ceil(static_cast<double>(padding.padded.W) / padding.reduced.W);
        unsigned y_coef = std::ceil(static_cast<double>(padding.padded.H) / padding.reduced.H);

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

            /*Select best MPE mode*/
            if(padding.mode.H == padding.mode.W)
                workload.MPEMode = mv::Matrix;
            else if ((padding.mode.H == 1) && (padding.mode.W == 16))
                workload.MPEMode = mv::MPE_Mode::Vector;
            else if ((padding.mode.W == 1) && (padding.mode.H == 16))
                workload.MPEMode = mv::MPE_Mode::Vector;
            else
                workload.MPEMode = mv::MPE_Mode::Vector_FP16;

            /*store requested number of workoads*/
            workload.requestedWorkloadNumber = nWorkloads;
            workload.algorithm = "Rectangle";

            workload_list.push_back(workload);
        }

        return workload_list;
    }

} // namespace mv[]


int mv::Workloads::partitionTensorWithRectangleHeuristic(const mv::DPUModeList& mode_list,
                                                                   size_t        nWorkloads,
                                                                   bool         split_over_h,
                                                                   bool         split_over_w,
                                                                   bool         split_symmetric,
                                                         const mv::WorkloadSplitMode& split_mode,
                                                         const mv::pass::PassEntry &pass)
{
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: layer=" + layerName_);

    unsigned C, H, W, N;
    if (tensorShape_.ndims() < 4) {
        pass.log(mv::Logger::MessageType::Error,
                 "RectangleHeuristic: too few tensor ndims=" + std::to_string(tensorShape_.ndims()));
        return -4;
    }
    W = tensorShape_[IO_WIDTH_DIMENSION];
    H = tensorShape_[IO_HEIGHT_DIMENSION];
    C = tensorShape_[IO_CHANNEL_DIMENSION];
    N = tensorShape_[IO_BATCH_DIMENSION];
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: height=" + std::to_string(H)
                                                              + ", width=" + std::to_string(W));

    WorkloadShape original_shape;
    original_shape.H = H; // height,aka Y
    original_shape.W = W; // width, aka X

    // enable splitting over Z
    if (split_mode == mv::WorkloadSplitMode::HC)
    {
        original_shape.W = C;
    }
    else if (split_mode == mv::WorkloadSplitMode::WC)
    {
        original_shape.H = C;
    }
    else if (split_mode == mv::WorkloadSplitMode::NC || split_mode == mv::WorkloadSplitMode::C)
    {
        original_shape.H = N;
        original_shape.W = C;
    }
    if (original_shape.H == 0 || original_shape.W == 0)
    {
        pass.log(mv::Logger::MessageType::Error,
                 "RectangleHeuristic: invalid shape: W=" + std::to_string(original_shape.W) +
                                                  ", H=" + std::to_string(original_shape.H));
        return -4;
    }

    auto best_padding = selectPadding(original_shape, mode_list);
    auto& reduced_shape = best_padding.reduced;
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: reduced_height=" + std::to_string(reduced_shape.H)
                                                             + ", reduced_width="  + std::to_string(reduced_shape.W));

    SplitSliceVariant slicing_variant = splitSliceSymmetric(reduced_shape.W, reduced_shape.H, nWorkloads,
                                                            split_over_h, split_over_w);
    if (!split_symmetric)
    {
        SplitSliceVariant slicing_variant_2 = splitSliceNonSymmetric(reduced_shape.W, reduced_shape.H, nWorkloads,
                                                                     split_over_h, split_over_w);
        if (slicing_variant.cost_estimate > slicing_variant_2.cost_estimate)
            slicing_variant = slicing_variant_2;
    }
    if (std::isinf(slicing_variant.cost_estimate))
    {
        pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: cannot slice!");
        return -4;
    }
    SplitSliceList& slice_list = slicing_variant.slice_list;
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: slices=" + std::to_string(slice_list.size()));

    unsigned Z = C;
    if (split_mode == mv::WorkloadSplitMode::HC || split_mode == mv::WorkloadSplitMode::C)
    {
        Z = W;
    }
    else if (split_mode == mv::WorkloadSplitMode::WC)
    {
        Z = H;
    }
    else if (split_mode == mv::WorkloadSplitMode::NC)
    {
        Z = W;
    }


    workloads_ = generateWorkloadsFromSlices(mode_list, slice_list, best_padding, nWorkloads, Z);
    pass.log(mv::Logger::MessageType::Debug, "RectangleHeuristic: done");


    return 1;
}

const std::vector<mv::Workload>& mv::Workloads::overlap_and_clip(std::array<unsigned short, 4> &padding, const Shape &tensorShape)
{
    for (auto workload = workloads_.begin(); workload != workloads_.end(); workload++)
    {
        workload->MinY = std::max((workload->MinY - padding[2]), 0);
        workload->MinX = std::min((workload->MinX - padding[0]), 0);
        workload->MaxY = std::min((workload->MaxY + padding[3]), int (tensorShape[IO_HEIGHT_DIMENSION] - 1));
        workload->MaxX = std::max((workload->MaxX + padding[1]), int(tensorShape[IO_WIDTH_DIMENSION] - 1));
    }
    return workloads_;
}

void mv::Workloads::add_xy_offset(std::vector<std::size_t>& offset)
{
    for (auto workload = workloads_.begin(); workload != workloads_.end(); workload++)
    {
        workload->MinY = workload->MinY + offset[IO_HEIGHT_DIMENSION];
        workload->MinX = workload->MinX + offset[IO_WIDTH_DIMENSION];
        workload->MaxY = workload->MaxY + offset[IO_HEIGHT_DIMENSION];
        workload->MaxX = workload->MaxX + offset[IO_WIDTH_DIMENSION];
    }

}

void mv::Workloads::apply_z_offset(std::vector<std::size_t>& offset)
{
    for (auto workload = workloads_.begin(); workload != workloads_.end(); workload++)
    {
        workload->z_offset += offset[IO_CHANNEL_DIMENSION];
        auto workloadOutputChannels = workload->MaxZ - workload->MinZ;
        workload->MinZ = workload->z_offset;
        workload->MaxZ = workload->z_offset + workloadOutputChannels;
    }

}

void mv::Workloads::populateClusterID(int clusterID)
{
    for (auto workload = workloads_.begin(); workload != workloads_.end(); workload++)
        workload->clusterID = clusterID;
}

int mv::Workloads::partitionTensorWithZsplit(const mv::DPUModeList& mode_list, size_t nWorkloads, const mv::pass::PassEntry &pass)
{
    pass.log(mv::Logger::MessageType::Debug, "Zsplit: layer=" + layerName_);
    unsigned C, H, W;
    if (tensorShape_.ndims() < 2) {
        pass.log(mv::Logger::MessageType::Error,
                 "Zsplit: too few tensor ndims=" + std::to_string(tensorShape_.ndims()));
        return -4;
    }

    W = tensorShape_[0];
    H = tensorShape_[1];
    C = tensorShape_.ndims() >= 3 ? tensorShape_[2] : 0;

    //max Z calculation
    size_t max_channels_per_WL = divRoundUp(C,nWorkloads);
    if (max_channels_per_WL < 16)
        return 0;

    WorkloadShape original_shape;
    original_shape.W = W; // width, aka X
    original_shape.H = H; // height,    Y
    auto best_padding = selectPadding(original_shape, mode_list);

    // split the output channels into per workload
    size_t output_channels = 0;
    for (size_t idx = 0; idx < nWorkloads; idx++)
    {
        Workload workload;
        output_channels = std::min<size_t>(max_channels_per_WL, C - idx*max_channels_per_WL);

        /*Populate the workloads*/
        workload.MinX = 0;
        workload.MinY = 0;
        workload.MaxX = W-1;
        workload.MaxY = H-1;

        workload.z_offset = idx*max_channels_per_WL;
        workload.MinZ = idx*max_channels_per_WL;
        workload.MaxZ = workload.MinZ + output_channels -1;
        workload.requestedWorkloadNumber = nWorkloads;
        workload.algorithm = "Z-Tiling";

        /*Select best MPE mode*/
        if(best_padding.mode.H == best_padding.mode.W)
            workload.MPEMode = mv::Matrix;
        else if ((best_padding.mode.H == 1) && (best_padding.mode.W == 16))
            workload.MPEMode = mv::MPE_Mode::Vector;
        else if ((best_padding.mode.W == 1) && (best_padding.mode.H == 16))
            workload.MPEMode = mv::MPE_Mode::Vector;
        else
            workload.MPEMode = mv::Vector_FP16;

        //Check that the z workloads dimension for z tiling are multiple of 16
        if(((workload.MaxZ+1) - workload.MinZ)%16 != 0)
            return 0;

        workloads_.push_back(workload);

    }

    pass.log(mv::Logger::MessageType::Debug, "Number of workloads with Zsplit =" + std::to_string(workloads_.size()));
    pass.log(mv::Logger::MessageType::Debug, "Zsplit: done");

    return 1;
}
