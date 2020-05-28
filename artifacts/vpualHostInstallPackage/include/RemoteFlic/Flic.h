///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      Flic.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for Host side FLIC usage over VPUAL.

#ifndef __FLIC_H__
#define __FLIC_H__

#include <string>
#include <stdint.h>
#include <list>

#include "VpualDispatcher.h"

#define MAX_PLUGS_PER_PIPE (32)

/** Forward declarations. */
class Message;
class PluginStub;

/**
 * Pipeline Stub Class.
 * This object looks like the real FLIC pipeline object on the device, but it is
 * a stub and all method calls are remote.
 */
class Pipeline : private VpualStub
{
  public:
    Pipeline(int maxPlugs = MAX_PLUGS_PER_PIPE);

    virtual void Add(PluginStub *plg, const char *name = NULL);

    virtual void Delete();
    virtual void Start ();
    virtual void Stop  ();
    virtual void Wait  ();
    int          Has   (PluginStub *plg);

  private:
    std::list<PluginStub *> plugins;
};

/**
 * Base class for all Plugin Stubs.
 * Similar to ThreadedPlugin or IPlugin in the FLIC framework.
 */
class PluginStub : public VpualStub
{
  private:
    uint32_t io_count = 0;

  public:
    /** Constructor just invokes the parent constructor. */
    PluginStub(std::string type) : VpualStub(type){};

    virtual void Stop  () {};   // By default do nothing.
    virtual void Delete() {};   // By default do nothing.
    virtual void Wait  () {};   // By default do nothing.

  protected:
    /** Add Messages to the Plugin (in "Create" function typically). */
    void Add(Message *s);
};

#endif // __FLIC_H__
