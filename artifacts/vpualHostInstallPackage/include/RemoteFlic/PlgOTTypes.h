#ifndef __PLG_OTTYPES_H__
#define __PLG_OTTYPES_H__
#include <vector>


namespace vpuot
{

struct Rect
{
  public:
    int32_t x, y, width, height, area;
    Rect() {
        x = y = width = height = area = 0;
    }
    Rect(int32_t x, int32_t y, int32_t width, int32_t height) : x(x), y(y), width(width), height(height) {
        width = std::max((int32_t) 0, width);
        height = std::max((int32_t) 0, height);
        area = width * height;
    }
};

enum TrackType
{
    ZEROTERM = 0,
    SHORTTERM = 1
};


/**
 * @enum TrackingStatus
 *
 * Tracking status.
 */
enum struct TrackingStatus
{
    NEW,         /**< The object is newly added. */
    TRACKED,     /**< The object is being tracked. */
    LOST         /**< The object gets lost now. The object can be tracked again automatically(long term tracking) or by specifying detected object manually(short term and zero term tracking). */
};

/**
 * @class DetectedObject
 * @brief Represents an input object.
 *
 * In order to track an object, detected object should be added one or more times to ObjectTracker.
 * When an object is required to be added to ObjectTracker, you can create an instance of this class and fill its values.
 */

class DetectedObject
{
public:
    DetectedObject(const Rect& input_rect, int32_t input_class_label) : rect(input_rect), class_label(input_class_label) {}

    Rect rect;
    int32_t class_label = -1;
};
typedef std::vector<DetectedObject> DetectedObjects;

/**
 * @struct Object
 * @brief Represents tracking result of a target object.
 *
 * It contains tracking information of a target object.
 * ObjectTracker generates an instance of this class per tracked object when Track method is called.
*/

class Object {
public:
    /**

     * Object rectangle.
     */
    Object() : rect(Rect()), tracking_id (-1), class_label(-1), status(TrackingStatus::LOST)  {}
    Object(const Rect& rect, uint64_t tid, int32_t class_label, TrackingStatus status) : rect(rect), tracking_id(tid), class_label(class_label), status(status) {}
    Rect rect;

    /**
     * Tracking ID.
     */
    uint64_t tracking_id;

    /**
     * Class label.
     * It is the value specified in DetectedObject.
     */
    int32_t class_label;

    /**
     * Tracking status.
     */
    TrackingStatus status;

};

typedef std::vector<Object> Objects;
struct OutObjects {} ;
#if 0
struct OutObjects : public PoBuf {
    public:
        Object *objects;
        uint32_t nObjects;
        uint32_t channelID;
        OutObjects() {
            channelID=0;
            objects = NULL;
            nObjects = 0;
        }
};
#endif
typedef PoPtr<OutObjects> OutObjectsPtr;

};  // namespace vpuot

#endif  // __PLG_OTTYPES_H__
