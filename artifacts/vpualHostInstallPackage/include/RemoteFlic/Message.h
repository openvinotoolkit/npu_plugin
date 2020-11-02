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
/// @file      Message.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for FLIC message types on Host over VPUAL.
///

#ifndef __MESSAGE_H__
#define __MESSAGE_H__

#include "VpualDispatcher.h"
#include "Flic.h"
#include <iostream>

// TODO - we have a decoder for each Sender which isn't actually attached to
// the real message object. We could just do one global decoder since we are
// finding the messages through the id's of their parent plugins, or we could
// find a way to attach the real message objects to the decoders...

/**
 * Message stub base class.
 * All FLIC message stub types inherit from this class.
 */
class Message
{
  public:
    uint32_t io_id; // TODO may rework this.
    // Pointer to parent plugin
    PluginStub *parent;
};

/** Slave reciever template class. */
template <typename T>
class SReceiver : public Message
{
};

/** Master reciever template class. */
template <typename T>
class MReceiver : public Message
{
};

/**
 * Master sender template class.
 * Contains Link method to Link with a compatable SReceiver.
 */
template <typename T>
class MSender : private VpualStub, public Message
{
  public:
    /** Constructor declaration (definition is type dependant). */
    MSender(uint32_t device_id);

    // TODO - May be gcc bug, but we need this declaration to help with initialisation.
    //        Copy-elision should occur, so we will never use it.
    MSender(const MSender&); // Declare copy ctor, but don't define.

    /**
     * Link to a compatable slave receiver.
     *
     * @param r slave reciever of the same type (T) to Link with.
     */
    void Link(SReceiver<T> *r)
    {
        if(!r) throw std::invalid_argument("Argument must not be NULL");
        if(!(r->parent)) throw std::invalid_argument("Argument parent must not be NULL");

        // Command message.
        VpualMessage cmd(64);

        // Check that messages have parents:
        assert(this->parent);

        cmd.serialize(&this->parent->stubID, sizeof(this->parent->stubID)); // ID of output plugin
        cmd.serialize(&this->io_id, sizeof(this->io_id));                   // ID of IO in this plugin
        cmd.serialize(&r->parent->stubID, sizeof(r->parent->stubID));       // ID of other plugin
        cmd.serialize(&r->io_id, sizeof(r->io_id));                         // ID of IO in other plugin

        // Dispatch command.
        VpualDispatch(&cmd, NULL);
    }
};

/**
 * Slave receiver template class.
 * Contains Link method to Link with a compatable MReceiver.
 */
template <typename T>
class SSender : private VpualStub, public Message
{
  public:
    /** Constructor declaration (definition is type dependant). */
    SSender(uint32_t device_id);

    /**
     * Link to a compatable master receiver.
     *
     * @param r master reciever of the same type (T) to Link with.
     */
    void Link(MReceiver<T> *r)
    {
        if(!r) throw std::invalid_argument("Argument must not be NULL");
        if(!(r->parent)) throw std::invalid_argument("Argument parent must not be NULL");

        // Command message.
        VpualMessage cmd(64);

        // Check that messages have parents:
        assert(this->parent);

        cmd.serialize(&this->parent->stubID, sizeof(this->parent->stubID)); // ID of output plugin
        cmd.serialize(&this->io_id, sizeof(this->io_id));                   // ID of IO in this plugin
        cmd.serialize(&r->parent->stubID, sizeof(r->parent->stubID));       // ID of other plugin
        cmd.serialize(&r->io_id, sizeof(r->io_id));                         // ID of IO in other plugin

        // Dispatch command.
        VpualDispatch(&cmd, NULL);
    }
};

/** Pool sender template class (simply SSender renamed). */
template <typename P>
class PoolSender : public SSender<P>
{
	public:
    PoolSender(uint32_t device_id) : SSender<P>(device_id) {}
};

#endif //__MESSAGE_H__
