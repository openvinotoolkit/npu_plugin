#ifndef MV_WORKLOAD_STRUCT
#define MV_WORKLOAD_STRUCT
#include <cstdint>
#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <functional>

namespace mv
{
    enum MPE_Mode
    {
        Vector,
        Matrix,
        Vector_FP16
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
        int16_t padLeft = 0;
        int16_t padRight = 0;
        int16_t padTop = 0;
        int16_t padBottom = 0;
        int32_t clusterID = 0;
        int8_t workloadID = 0;
        int16_t z_offset = 0;
        int16_t requestedWorkloadNumber = 0;
        std::string algorithm = "None";

        int16_t area()
        {
          return (MaxX - MinX + 1) * (MaxY - MinY + 1);
        }
        std::vector<std::pair<int16_t, int16_t>> points;
        int16_t pointsTotal()
        {
         return points.size();
        }
        std::vector<std::pair<int16_t, int16_t>> vertices;

        void setVertices()
        {
            vertices.push_back(std::make_pair(MinX,MinY));
            vertices.push_back(std::make_pair(MinX,MaxY));
            vertices.push_back(std::make_pair(MaxX,MinY));
            vertices.push_back(std::make_pair(MaxX,MaxY));
        }
        void setMinMaxAndVertices()
        {
            MinX = INT16_MAX;
            MaxX = 0;
            MinY = INT16_MAX;
            MaxY = 0;
            for(auto it = points.begin(); it != points.end(); it++)
            {
                MinX = std::min(MinX,it->first);
                MaxX = std::max(MaxX, it->first);
                MinY = std::min(MinY,it->second);
                MaxY = std::max(MaxY, it->second);
            }
            setVertices();
        }
    };
}
#endif
