#ifndef _PRE_PROC_API_H_
#define _PRE_PROC_API_H_

#include <stdio.h>
#include "PlgPreProc.h"
#include "PlgXlinkIn.h"
#include "PlgXlinkOut.h"
#include "PlgPreProcXIn.h"

#include <vpumgr.h>
#include "cma_allocation_helper.h"
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include <thread_queue.h>

#include <semaphore.h>

#include <thread>

#ifndef MAX_PLUGS_PER_PIPE
#define MAX_PLUGS_PER_PIPE (32)
#endif

typedef struct
{
    sem_t sem;
    uint32_t inputY;   //phy address
    uint32_t inputUV;   //phy address
    uint32_t output;  //phy address
    CmaData *desc; //PreProcDesc
    frameSpec inspecY;
    frameSpec inspecUV;
    frameSpec outspec;
} preproc_req;

class VPUPreProc {
public:
    VPUPreProc(const uint32_t device_Id, const int32_t requests_Num = 16): deviceId(device_Id),
		reqsNum(requests_Num)
		{};
    virtual ~VPUPreProc() {};

    int32_t Init();
    int32_t Process(uint32_t paddr_inputY, uint32_t paddr_inputUV, uint32_t paddr_output, PreProcDesc desc, frameSpec inspecY, frameSpec inspecUV, frameSpec outspec);
    int32_t Destroy();
    int32_t GetIdleRequest(preproc_req ** req);

    CThreadQueue<preproc_req *> reqs_idle;
    CThreadQueue<preproc_req *> reqs_forward;
    CThreadQueue<preproc_req *> reqs_processing;

    std::shared_ptr<Pipeline> pipe;
    std::shared_ptr<PlgXlinkIn> plgSrcY;
    std::shared_ptr<PlgXlinkIn> plgSrcUV;
    std::shared_ptr<PlgXlinkIn> plgSrcObuf;
    std::shared_ptr<PlgXlinkOut> plgSink;
    std::shared_ptr<PlgPreProc> plgPreProc;
    std::shared_ptr<PlgPreProcXIn> plgPPCfg;

private:
    uint32_t deviceId;
    int32_t reqsNum;
    preproc_req *ppRequests;

    std::thread inputThread;
    std::thread outputThread;

    void initVpualObjects(const uint32_t device_Id);
};


#endif
