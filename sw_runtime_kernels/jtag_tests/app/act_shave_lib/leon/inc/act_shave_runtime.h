#pragma once

#include <descriptor.h>

int  startActShave(void);
void stopActShave(void);
bool enqueueShaveJob(shv_job_header * job);