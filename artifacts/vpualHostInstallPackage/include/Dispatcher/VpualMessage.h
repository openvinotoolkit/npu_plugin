///
/// @file      VpualMessage.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for VPUAL Message Class.
///
#ifndef __VPUAL_MESSAGE_H__
#define __VPUAL_MESSAGE_H__

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Vpual Command Header Structure.
 */
// TODO this doesn't necessarily need to be in this header, but needs to be shared between host and device.
// TODO the names and types can be better here.
typedef struct
{
    /**
	 * Type of message.
	 */
    enum messageType
    {
        DECODER_CREATE = 1,  /*< Create a decoder. */
        DECODER_DESTROY = 2, /*< Destroy a decoder. */
        DECODER_DECODE = 3,  /*< Call a decoder's custom "Decode" method. */
    };

    uint32_t magic;    /*< Magic value. */
    uint32_t checksum; /*< Checksum of payload? */
    uint32_t msg_id;   /*< ID of message. */

    uint32_t length;  /*< Length of payload. */
    uint32_t stubID;  /*< ID of stub/decoder. */
    messageType type; /*< Type of message. */
} VpualCmdHeader_t;

/**
 * VpualMessage Class
 *
 * This class handles serialization and deserialization (read/write) from a buffer.
 * The buffer can be allocated by the class, or it can be provided to the class.
 * If the buffer is provided to the object then the creator must ensure that the
 * buffer exists for the lifetime of the object.
 */
// TODO unsure about copy-constructors, assignments etc. Read only version (no write)?
// Could use smart pointers to simplify the object's existance?
class VpualMessage
{
  private:
    uint32_t rp = 0; /*< Read position. */
    uint32_t wp = 0; /*< Write position. */
    uint32_t length; /*< Message data length. */
    uint8_t *sdata;  /*< Message data in serial format. */

    /** Deallocate memory on destruction? */
    bool deallocate = false;

  public:
    /**
	 * Create a buffer based on provided length.
	 *
	 * @param len the length (bytes) of the buffer to create.
	 */
    void create(const uint32_t len);

    /**
	 * Constructor without buffer.
	 *
	 * Construct the object without a buffer. The buffer must be created later.
	 */
    VpualMessage() : length(0), sdata(NULL) {}

    /**
	 * Constructor with allocation.
	 * Construct the object and allocate a buffer on the heap.
	 *
	 * @param len the length (bytes) of the buffer to create.
	 */
    VpualMessage(const uint32_t len) : length(len), sdata(NULL)
    {
        create(len);
    }

    /**
	 * Constructor with buffer provided.
	 * Construct the object and using the provided buffer.
	 *
	 * @param data pointer to the provided buffer.
	 * @param len the length (bytes) of the buffer to create.
	 */
    VpualMessage(uint8_t *const data, const uint32_t len) : length(len), sdata(data)
    {
        assert(data || (len == 0)); // Ensure not NULL unless length is 0.
    }

    /**
	 * Destructor.
	 * Will destroy a buffer if it was created by the object.
	 */
    ~VpualMessage()
    {
        if (deallocate)
        {
            delete sdata;
        }
    }

    // Delete copy constructor and assignment operator.
    VpualMessage(const VpualMessage&) = delete;
    VpualMessage& operator=(const VpualMessage&) = delete;

    /**
	 * Serialize data onto the buffer.
	 *
	 * @param idata pointer to data to be written onto the buffer.
	 * @param size length of data (bytes) to be written onto the buffer.
	 */
    int serialize(const void *const idata, const uint32_t size);

    /**
	 * Deserialize data from the buffer.
	 *
	 * @param odata pointer to destination of deserailized data.
	 * @param size length of data (bytes) to be read from the buffer.
	 */
    int deserialize(void *const odata, const uint32_t size);

    /**
	 * Get length of serialized data.
	 *
	 * @return data length.
	 */
    uint32_t len(void) const { return wp; }

    /**
	 * Get the buffer.
	 *
	 * @return start address of buffer.
	 */
    uint8_t *dat(void) const { return sdata; }

    /**
     * Set a new write position for the buffer.
     *
     * @param new_wp the new write position (bytes) for the buffer.
     */
    void set_len(uint32_t new_wp) {
        wp = new_wp;
    }
};

#endif // __VPUAL_MESSAGE_H__
