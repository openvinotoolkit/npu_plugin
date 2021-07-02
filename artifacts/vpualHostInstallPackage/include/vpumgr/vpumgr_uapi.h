// {% copyright %}
/* SPDX-License-Identifier: GPL-2.0+ WITH Linux-syscall-note
 * VPU manager Linux Kernel API
 *
 */
#ifndef __VPUMGR_UAPI_H
#define __VPUMGR_UAPI_H

#include <linux/types.h>

/* ioctl numbers */
#define VPUMGR_MAGIC 'V'
/* VPU manager IOCTLs */
#define VPUMGR_IOCTL_DMABUF_ALLOC _IOWR(VPUMGR_MAGIC, 2, struct vpumgr_args_alloc)
#define VPUMGR_IOCTL_DMABUF_IMPORT _IOWR(VPUMGR_MAGIC, 3, struct vpumgr_args_import)
#define VPUMGR_IOCTL_DMABUF_UNIMPORT _IOWR(VPUMGR_MAGIC, 4, __s32)
#define VPUMGR_IOCTL_DMABUF_PTR2VPU _IOWR(VPUMGR_MAGIC, 5, __u64)
#define VPUMGR_IOCTL_VCM_SUBMIT _IOWR(VPUMGR_MAGIC, 6, struct vpumgr_vcm_submit)
#define VPUMGR_IOCTL_VCM_WAIT _IOWR(VPUMGR_MAGIC, 7, struct vpumgr_vcm_wait)
#define VPUMGR_IOCTL_FETCH_META _IOWR(VPUMGR_MAGIC, 8, struct vpumgr_args_fetch_meta)
#define VPUMGR_IOCTL_DMABUF_SYNC _IOWR(VPUMGR_MAGIC, 9, struct vpumgr_args_sync)
#define VPUMGR_IOCTL_END _IO(VPUMGR_MAGIC, 10)

struct vpumgr_args_alloc {
    __s32 fd;          /* out: DMABuf fd */
    __s32 reserved;    /*  in: reserved */
    __s32 noncoherent; /*  in: noncoherent */
    __u64 size;        /*  in: required buffer size */
};

/* vpu_access flags */
enum vpu_access_type { VPU_ACCESS_DEFAULT = 0, VPU_ACCESS_READ = 1, VPU_ACCESS_WRITE = 2, VPU_ACCESS_RW = 3 };

struct vpumgr_args_import {
    __s32 fd;         /*  in: input DMABuf fd */
    __s32 vpu_access; /*  in: how vpu is going to access the buffer */
    __u64 vpu_addr;   /* out: vpu dma address of the DMABuf */
    __u64 size;       /* out: the size of the DMABuf */
};

/*args to be exchanged between the client and driver for vpurm_fetch_meta_from_fd */
struct vpumgr_args_fetch_meta {
    __s32 fd;          /*  in: input DMABuf fd */
    __u64 size;        /*  in: the size of struct _VIV_VIDMEM_METADATA */
    void *meta_buffer; /*out: buffer pointer to get VSI Meta Data */
};

/* Command code reserved for kernel mode driver,
 * user-space should not use commmand code smaller
 * than or equal to this micro
 */
#define VCTX_KMD_RESERVED_CMD_LAST 31

struct vpumgr_vcm_submit {
    __u32 cmd;       /*  in: command code */
    __u64 in;        /*  in: input paramer buffer address */
    __u32 in_len;    /*  in: input paramer buffer length */
    __s32 submit_id; /* out: submit id */
};

struct vpumgr_vcm_wait {
    __s32 submit_id;  /*  in: submit id */
    __s32 vpu_rc;     /* out: vpu return code */
    __u64 out;        /*  in: address of the buffer for receiving result */
    __u32 out_len;    /*  in: length of the result buffer */
    __u32 timeout_ms; /*  in: timeout in milliseconds */
};

#endif /* __VPUMGR_UAPI_H */
