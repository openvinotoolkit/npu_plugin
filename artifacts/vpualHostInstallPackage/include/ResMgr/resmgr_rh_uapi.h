/*GPL-2.0-only */
/*
 * VPU resource manager Linux Kernel API
 *
 * Copyright (C) 2018-2019 Intel Corporation
 *
 */
#ifndef __RESMGR_RH_UAPI_H
#define __RESMGR_RH_UAPI_H

#define RESMGR_MAGIC       'V'
#define CHANNEL_NAME_LEN_MAX   (40)

#include <linux/types.h>

enum ALLOC_CHANNEL_TYPE {
    PCIE_CHANNEL = 1,
    END_CHANNEL_TYPE
};

struct resmgr_args_cm_ctx {
    __u16 channel;
    __u16 name_len;
    enum ALLOC_CHANNEL_TYPE type;
    char chan_name[CHANNEL_NAME_LEN_MAX];
    uint32_t sw_device_id;
};

#define RESMGR_IOCTL_CM_GET_CHANNEL             _IOWR(RESMGR_MAGIC, 0, struct resmgr_args_cm_ctx)
#define RESMGR_IOCTL_CM_PUT_CHANNEL             _IOWR(RESMGR_MAGIC, 1, struct resmgr_args_cm_ctx)
#define RESMGR_IOCTL_END                   	    _IO(RESMGR_MAGIC, 2)

#endif /* __RESMGR_RH_UAPI_H */
