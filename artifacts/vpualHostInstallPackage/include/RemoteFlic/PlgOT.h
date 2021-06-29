// {% copyright %}
///
/// @file      PlgOT.h
///

#ifndef _PLG_OT_H_
#define _PLG_OT_H_
//#include <ImgFrame.h>
#include <stdio.h>
#include <vector>
#include <map>
#include "Flic.h"
#include "Message.h"
#include "Pool.h"
#include "PlgOTTypes.h"

class PlgOT : public PluginStub{
  public:
    PlgOT(uint32_t device_id = 0) : PluginStub("PlgOT", device_id), out{device_id} {}


    SReceiver<ImgFramePtr> in0;
    SReceiver<vpuot::DetectedObjects> in1;
    SReceiver<float> in2;
    MReceiver<vpuot::OutObjectsPtr> in3;

    MSender<vpuot::OutObjectsPtr> out;

    // Support for backward compatibility.
    /**
     * Create OT plugin
     *
     * @param ot_type                 - Tracking type for newly created ObjectTracker instance.
     * @param max_objects             - Maximum number of trackable objects in a frame.
     * @param mask_padding_thickness  - deprecated.
     * @retval int                    - Return the number of allocated shaves if Success, otherwise -1
     */
    int32_t  Create(vpuot::TrackType ot_type, int32_t max_objects, float mask_padding_thickness);


    /**
     * Create OT plugin
     *
     * @param ot_type                 - Tracking type for newly created ObjectTracker instance.
     * @param max_objects             - Maximum number of trackable objects in a frame.
     * @param mask_padding_thickness  - deprecated.
     * @param num_shaves              - The number of shaves are used on KMB
     * @param debugging_info          - The map with debugging params
     * @retval int                    - Return the number of allocated shaves if Success, otherwise -1
     */
    int32_t  Create(vpuot::TrackType ot_type, int32_t max_objects, float mask_padding_thickness, int32_t num_shaves, std::map<std::string, std::string>& debugging_info);


    /**
     * Create OT plugin
     *
     * @param ot_type                 - Tracking type for newly created ObjectTracker instance.
     * @param max_objects             - Maximum number of trackable objects in a frame.
     * @param tracking_per_class      - Tracker uses detection class or not.
     * @param num_allocated_shaves    - The number of allocated shaves
     * @param ot_debug_buf_addr       - Physical address for ot debug buffer
     * @param ot_debug_buf_size       - Size of ot debug buffer
     * @retval vpuot::ot_status_code  - Status of the creation (OT_SUCCESSFUL is success).
     */
    vpuot::ot_status_code  Create(vpuot::TrackType ot_type, int32_t max_objects, bool tracking_per_class, int32_t* num_allocated_shaves, uint64_t ot_debug_buf_addr = 0, uint32_t ot_debug_buf_size = 0);


    /**
     * Create OT plugin
     *
     * @param ot_type                 - Tracking type for newly created ObjectTracker instance.
     * @param max_objects             - Maximum number of trackable objects in a frame.
     * @param tracking_per_class      - Tracker uses detection class or not.
     * @param num_shaves              - The number of shaves are used on KMB
     * @param debugging_info          - The map with debugging params
     * @param num_allocated_shaves    - The number of allocated shaves
     * @param ot_debug_buf_addr       - Physical address for ot debug buffer
     * @param ot_debug_buf_size       - Size of ot debug buffer
     * @retval vpuot::ot_status_code  - Status of the creation (OT_SUCCESSFUL is success).
     */
    vpuot::ot_status_code  Create(vpuot::TrackType ot_type, int32_t max_objects, bool tracking_per_class, int32_t num_shaves, std::map<std::string, std::string>& debugging_info, int32_t* num_allocated_shaves, uint64_t ot_debug_buf_addr = 0, uint32_t ot_debug_buf_size = 0);

};
#endif
