// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

using Resources = int;

#define BBOX_MIN_MATCH_RATIO (0.5f)
#define LANDMARK_MAX_DISTANCE (15)

namespace {

std::string PVA_MTCNN_MODEL = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
    <layers>
        <layer id="0" name="data" precision="FP16" type="Input">
            <output>
                <port id="0">
                    __DIMS__
                </port>
            </output>
        </layer>
        <layer id="1" name="Add_" precision="FP16" type="ScaleShift">
            <input>
                <port id="0">
                    __DIMS__
                </port>
            </input>
            <output>
                <port id="3">
                    __DIMS__
                </port>
            </output>
            <blobs>
                <weights offset="0" size="6"/>
                <biases offset="6" size="6"/>
            </blobs>
        </layer>
        <layer id="2" name="custom_fd" precision="FP16" type="MTCNN">
            <data   pnet_ir="__PNET_MODEL_PATH__"
                    rnet_ir="__RNET_MODEL_PATH__"
                    onet_ir="__ONET_MODEL_PATH__"
                    pnet_resolutions="__PNET_RESOLUTIONS__"
                    mode="__MODE__"/>
            <input>
                <port id="0">
                    __DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>32</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>

)V0G0N";


#define AVA_FD "AVA_FaceDetector"
#define PUBLIC_MTCNN "PUBLIC_MTCNN"

class Pos {
public:
    Pos(): x(0), y(0) {}
    Pos(int _x, int _y) : x(_x), y(_y) {}
    Pos(const Pos& c) : x(c.x), y(c.y) {}
public:
    int x;
    int y;
};

class Rect {
public:
//    Rect():x(0),y(0),width(0),height(0) {}
    Rect():topleft(Pos(0, 0)),width(0),height(0) {}
    Rect(int _x, int _y, int _w, int _h):topleft(_x, _y), width(_w), height(_h) {}
    Rect(Pos _topleft, int _w, int _h):topleft(_topleft), width(_w), height(_h) {}
    Rect(const Rect& r):topleft(r.topleft),width(r.width), height(r.height) {}
    int area() const { return width * height; }
public:
//    int x;
//    int y;
    Pos topleft;
    int width;
    int height;
};

class Face {
public:
    Face(): bbox(Rect(0, 0, 0, 0)),
            left_eye(Pos(0, 0)),
            right_eye(Pos(0, 0)),
            nose(Pos(0, 0)),
            mouth_left_end(Pos(0, 0)),
            mouth_right_end(Pos(0, 0)) {}
    Face(Rect _bbox, Pos _left_eye, Pos _right_eye, Pos _nose, Pos _mouth_left_end, Pos _mouth_right_end):
            bbox(_bbox),
            left_eye(_left_eye),
            right_eye(_right_eye),
            nose(_nose),
            mouth_left_end(_mouth_left_end), mouth_right_end(_mouth_right_end) {}
    Face(const Face& f) :
            bbox(f.bbox),
            left_eye(f.left_eye),
            right_eye(f.right_eye),
            nose(f.nose),
            mouth_left_end(f.mouth_left_end),
            mouth_right_end(f.mouth_right_end) {}

public:
    Rect bbox;
    Pos left_eye;
    Pos right_eye;
    Pos nose;
    Pos mouth_left_end;
    Pos mouth_right_end;
};

static inline Rect GetIntersectionArea(const Rect& a, const Rect& b )
{
    int x1 = std::max(a.topleft.x, b.topleft.x);
    int y1 = std::max(a.topleft.y, b.topleft.y);
    int x2 = std::min(a.topleft.x + a.width,  b.topleft.x + b.width);
    int y2 = std::min(a.topleft.y + a.height, b.topleft.y + b.height);
    if( x2 <= x1 || y2 <= y1 )
        return Rect();

    Rect c;
    c.topleft.x = x1;
    c.topleft.y = y1;
    c.width = x2 - x1;
    c.height = y2 - y1;
    return c;
}

static std::vector<Face> AVA_gt_270_480_faces = {
    //   bbox                     left_eye          right_eye         nose               mouth_left_end    mouth_right_end
    Face(Rect( 209, -11, 53 , 57), Pos( 225,     7), Pos(242,     3 ), Pos(  232,    18), Pos( 231,    31), Pos(  243,    29)),
    Face(Rect( 286,  55, 49 , 49), Pos( 300,    74), Pos(315,    67 ), Pos(  306,    80), Pos( 307,    92), Pos(  319,    88)),
    Face(Rect( 369, 218, 55 , 58), Pos( 389,   233), Pos(408,   239 ), Pos(  391,   249), Pos( 383,   255), Pos(  402,   262)),
    Face(Rect( 107,  25, 44 , 44), Pos( 123,    36), Pos(139,    42 ), Pos(  126,    48), Pos( 117,    53), Pos(  132,    59)),
    Face(Rect( 308, 174, 45 , 52), Pos( 317,   193), Pos(334,   191 ), Pos(  322,   206), Pos( 321,   213), Pos(  336,   211)),
    Face(Rect(  16, 179, 47 , 51), Pos(  30,   194), Pos( 49,   197 ), Pos(   35,   208), Pos(  30,   215), Pos(   46,   218)),
    Face(Rect( 213, 167, 46 , 48), Pos( 229,   180), Pos(247,   184 ), Pos(  240,   194), Pos( 225,   199), Pos(  241,   203)),
    Face(Rect( 417, 124, 40 , 44), Pos( 430,   137), Pos(447,   142 ), Pos(  436,   151), Pos( 425,   153), Pos(  441,   158)),
    Face(Rect( 120, 151, 45 , 48), Pos( 136,   165), Pos(154,   169 ), Pos(  146,   179), Pos( 130,   184), Pos(  147,   187))
};

static std::vector<Face> AVA_gt_1280_720_faces = {
    //   bbox                       left_eye        right_eye       nose            mouth_left_end  mouth_right_end
    Face(Rect( 558, -28, 144, 150), Pos( 599,  24), Pos( 649,  12), Pos( 620,  49), Pos( 617,  86), Pos( 649,  80)),
    Face(Rect( 983, 578, 157, 177), Pos(1041, 628), Pos(1092, 645), Pos(1048, 674), Pos(1023, 698), Pos(1073, 713)),
    Face(Rect( 766, 151, 128, 128), Pos( 802, 198), Pos( 844, 183), Pos( 820, 218), Pos( 820, 248), Pos( 851, 239)),
    Face(Rect( 286,  68, 120, 120), Pos( 332,  98), Pos( 374, 114), Pos( 338, 130), Pos( 316, 145), Pos( 355, 161)),
    Face(Rect( 572, 447, 119, 126), Pos( 615, 481), Pos( 660, 493), Pos( 644, 521), Pos( 601, 532), Pos( 644, 543)),
    Face(Rect(  53, 474, 117, 143), Pos(  82, 519), Pos( 130, 524), Pos(  92, 554), Pos(  82, 577), Pos( 122, 581)),
    Face(Rect( 320, 401, 123, 136), Pos( 364, 445), Pos( 410, 451), Pos( 394, 478), Pos( 353, 491), Pos( 399, 499)),
    Face(Rect(1114, 338, 106, 109), Pos(1147, 369), Pos(1194, 382), Pos(1163, 404), Pos(1139, 410), Pos(1180, 421)),
    Face(Rect( 820, 466, 122, 138), Pos( 849, 519), Pos( 894, 510), Pos( 863, 551), Pos( 859, 568), Pos( 899, 563))
};

static std::vector<Face> AVA_gt_1920_1080_faces = {
    //   bbox                      left_eye         right_eye          nose              mouth_left_end    mouth_right_end
    Face(Rect( 835, -38, 222, 221), Pos( 902,    39), Pos( 975,    20), Pos( 932,    76), Pos( 927,   128), Pos( 977,   119)),
    Face(Rect(  68, 716, 189, 213), Pos( 125,   783), Pos( 199,   792), Pos( 144,   839), Pos( 124,   866), Pos( 187,   875)),
    Face(Rect( 858, 676, 180, 181), Pos( 922,   725), Pos( 991,   740), Pos( 966,   782), Pos( 905,   800), Pos( 969,   815)),
    Face(Rect(1148, 225, 193, 193), Pos(1204,   297), Pos(1265,   275), Pos(1228,   328), Pos(1230,   372), Pos(1278,   358)),
    Face(Rect( 428, 103, 182, 181), Pos( 496,   149), Pos( 562,   174), Pos( 508,   195), Pos( 477,   220), Pos( 535,   243)),
    Face(Rect(1478, 880, 225, 227), Pos(1560,   936), Pos(1635,   964), Pos(1565,  1007), Pos(1539,  1029), Pos(1614,  1056)),
    Face(Rect(1237, 696, 183, 213), Pos(1272,   780), Pos(1339,   767), Pos(1298,   827), Pos(1288,   857), Pos(1348,   848)),
    Face(Rect( 478, 603, 189, 205), Pos( 545,   669), Pos( 614,   677), Pos( 591,   718), Pos( 532,   737), Pos( 601,   748)),
    Face(Rect(1672, 505, 155, 168), Pos(1725,   551), Pos(1793,   571), Pos(1747,   607), Pos(1706,   612), Pos(1773,   634))
};

//TODO: Currently, public MTCNN result below is temporal due to a regression issue, and will be updated once it is resolved.
static std::vector<Face> PUBLIC_gt_270_480_faces = {
    //   bbox                   left_eye         right_eye         nose             mouth_left_end   mouth_right_end
    Face(Rect(379, 216, 42, 56), Pos(390,   234), Pos(409,   240 ), Pos(392,   250), Pos(385,   257), Pos(402,   262)),
    Face(Rect(217, -29, 48, 73), Pos(226,     1), Pos(246,    -3 ), Pos(233,    14), Pos(233,    29), Pos(246,    26)),
    Face(Rect(113,  17, 38, 53), Pos(125,    36), Pos(142,    43 ), Pos(127,    50), Pos(118,    54), Pos(134,    61)),
    Face(Rect(117, 138, 42, 65), Pos(136,   167), Pos(155,   170 ), Pos(148,   182), Pos(132,   187), Pos(151,   190)),
    Face(Rect(310, 167, 39, 60), Pos(317,   195), Pos(336,   191 ), Pos(325,   206), Pos(321,   216), Pos(339,   212)),
    Face(Rect(213, 153, 38, 63), Pos(230,   182), Pos(248,   185 ), Pos(241,   197), Pos(226,   202), Pos(243,   204)),
    Face(Rect(293,  48, 43, 56), Pos(301,    75), Pos(317,    68 ), Pos(307,    83), Pos(308,    95), Pos(322,    90)),
    Face(Rect( 20, 171, 44, 61), Pos( 31,   196), Pos( 51,   198 ), Pos( 38,   210), Pos( 30,   218), Pos( 49,   219)),
    Face(Rect(418, 114, 37, 57), Pos(430,   139), Pos(449,   143 ), Pos(437,   153), Pos(426,   155), Pos(445,   159))
};

struct MTCNNInputShiftScales {
    float scale;
    float b_shift;
    float g_shift;
    float r_shift;
};

PRETTY_PARAM(mtcnn_input_scale_shifts, MTCNNInputShiftScales);

static inline void PrintTo(const MTCNNInputShiftScales& param, ::std::ostream* os)
{
    MTCNNInputShiftScales data = param;
    *os << "scale shift: " << data.scale
            << ", " << data.b_shift << ", " << data.g_shift << ", " << data.r_shift;
}

struct MTCNNTestParams {
    std::string mode;
    std::string pnet_path;
    std::string rnet_path;
    std::string onet_path;
    std::string pnet_resolutions;
};

PRETTY_PARAM(mtcnn_test_params, MTCNNTestParams);

static inline void PrintTo(const MTCNNTestParams& param, ::std::ostream* os)
{
    MTCNNTestParams data = param;
    *os << "scale shift: " << data.mode
            << ", " << data.pnet_path << ", " << data.rnet_path << ", " << data.onet_path << ", " << data.pnet_resolutions;
}

struct MTCNNInputData {
    SizeVector          dims;
    std::string         input_path;
    std::vector<Face>   gt_values;
};


PRETTY_PARAM(mtcnn_input_data, MTCNNInputData);

static inline void PrintTo(const MTCNNInputData& param, ::std::ostream* os)
{
    MTCNNInputData data = param;
    *os << "mtcnn_param: " << data.input_path;
    for (auto& dim : data.dims)
        *os  << ", " << dim;
}

struct MTCNNResources {
    Resources resources;
};

PRETTY_PARAM(mtcnn_resouce_constraints, MTCNNResources);

static inline void PrintTo(const MTCNNResources& param, ::std::ostream* os)
{
    MTCNNResources data = param;
    *os << data.resources;
}

inline double GetDistance(Pos p1, Pos p2) {
    double dst_x = (double)p2.x - (double)p1.x;
    double dst_y = (double)p2.y - (double)p1.y;
    return sqrt(dst_x*dst_x + dst_y*dst_y);
}

using MTCNNConcatTestParams = std::tuple<mtcnn_input_data, mtcnn_test_params, mtcnn_input_scale_shifts, mtcnn_resouce_constraints>;
class myriadMTCNNTests_nightly: public myriadLayerTestBaseWithParam<MTCNNConcatTestParams> {
public:
    float GetIoU(const Rect& rect1, const Rect& rect2)
    {
        const Rect inter = GetIntersectionArea(rect1, rect2);
        const int inter_size = inter.area();
        const int union_size = rect1.area() + rect2.area() - inter_size;

        return static_cast<float>(inter_size) / union_size;
    }

    bool CompareResult(const std::vector<Face> gt_faces, const std::vector<Face> detected_faces)
    {
        int matched_count = 0;
        const int target_count = gt_faces.size();
        for (int i = 0; i < detected_faces.size(); ++i)
        {
            const Face& detected_face = detected_faces[i];
            for (int j = 0; j < target_count; ++j)
            {
                const Rect& gt_face_bbox = gt_faces[j].bbox;
                float iou = GetIoU(detected_face.bbox, gt_face_bbox);

                if (iou > BBOX_MIN_MATCH_RATIO)
                {
                    //Compare landmarks
                    double distance = (double)LANDMARK_MAX_DISTANCE;
                    if((distance = GetDistance(detected_face.left_eye,  gt_faces[j].left_eye))  > LANDMARK_MAX_DISTANCE) {
                        std::cout<<"left eye of gt["<<j<<"]"<<" mismatches with detected face for "<< distance << std::endl;
                        break;
                    }
                    if(GetDistance(detected_face.right_eye, gt_faces[j].right_eye) > LANDMARK_MAX_DISTANCE) {
                        std::cout<<"right eye of gt["<<j<<"]"<<" mismatches with detected face for "<< distance << std::endl;
                        break;
                    }
                    if(GetDistance(detected_face.nose, gt_faces[j].nose) > LANDMARK_MAX_DISTANCE) {
                        std::cout<<"nose of gt["<<j<<"]"<<" mismatches with detected face for "<< distance << std::endl;
                        break;
                    }
                    if(GetDistance(detected_face.mouth_left_end,  gt_faces[j].mouth_left_end)  > LANDMARK_MAX_DISTANCE ) {
                        std::cout<<"mouth_left_end of gt["<<j<<"]"<<" mismatches with detected face for "<< distance << std::endl;
                        break;
                    }
                    if(GetDistance(detected_face.mouth_right_end, gt_faces[j].mouth_right_end) > LANDMARK_MAX_DISTANCE ) {
                        std::cout<<"mouth_right_end of gt["<<j<<"]"<<" mismatches with detected face for "<< distance << std::endl;
                        break;
                    }
                    matched_count++;
                    break;
                }
            }
        }
        bool passed = matched_count == target_count;
        if(!passed) {
            std::cout<< "Detected " << matched_count << " faces." << std::endl;
        }
        return passed;
    }

    std::string GenModel(const MTCNNTestParams& param, const SizeVector& dims)
    {
        std::string model = PVA_MTCNN_MODEL;
        std::ostringstream __DIMS__;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i != 0)
                __DIMS__ << "\n";
            __DIMS__ << "<dim>" << dims[i] << "</dim>";
        }
        REPLACE_WITH_STR(model, "__MODE__", param.mode);
        REPLACE_WITH_STR(model, "__DIMS__", __DIMS__.str());
        auto pnet_model_path = ModelsPath() + param.pnet_path;
        auto rnet_model_path = ModelsPath() + param.rnet_path;
        auto onet_model_path = ModelsPath() + param.onet_path;
        REPLACE_WITH_STR(model, "__PNET_MODEL_PATH__", pnet_model_path.str());
        REPLACE_WITH_STR(model, "__RNET_MODEL_PATH__", rnet_model_path.str());
        REPLACE_WITH_STR(model, "__ONET_MODEL_PATH__", onet_model_path.str());
        REPLACE_WITH_STR(model, "__PNET_RESOLUTIONS__", param.pnet_resolutions);

        return model;
    }

    // TODO fix Ptr
    TBlob<uint8_t>::Ptr GenWeights(const float scale, const float a, const float b, const float c)
    {
        int numWeights = 6;
        TBlob<uint8_t>::Ptr weights = make_shared_blob<uint8_t, const SizeVector>(Precision::U8, C, {numWeights * sizeof(ie_fp16)});
        weights->allocate();

        auto *weights_buffer = weights->buffer().as<PrecisionTrait<Precision::FP16>::value_type *>();
        weights_buffer[0] = PrecisionUtils::f32tof16(1 / 255.0 * scale);
        weights_buffer[1] = PrecisionUtils::f32tof16(1 / 255.0 * scale);
        weights_buffer[2] = PrecisionUtils::f32tof16(1 / 255.0 * scale);
        weights_buffer[3] = PrecisionUtils::f32tof16(a / 255.0 * scale);
        weights_buffer[4] = PrecisionUtils::f32tof16(b / 255.0 * scale);
        weights_buffer[5] = PrecisionUtils::f32tof16(c / 255.0 * scale);

        return weights;
    }

    void InitNetwork(const std::string& model, const TBlob<uint8_t>::Ptr weights_ptr, const Resources resources)
    {
        ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
        ASSERT_TRUE(_net_reader.isParseSuccess());

        _net_reader.SetWeights(weights_ptr);

        auto network = _net_reader.getNetwork();

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["data"]->setPrecision(Precision::U8);
        _inputsInfo["data"]->setLayout(Layout::NHWC);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["custom_fd"]->setPrecision(Precision::FP16);

        StatusCode st;
        if (resources != -1) {
            std::map<std::string, std::string> config;
            config["VPU_NUMBER_OF_SHAVES"] = std::to_string(resources);
            config["VPU_NUMBER_OF_CMX_SLICES"] = std::to_string(resources);
            ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, config, &_resp));
        } else {
            ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
        }

        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    void SetInput(const std::string& input_file)
    {
        StatusCode st;
        Blob::Ptr input;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("data", input, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        auto *input_buffer = input->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
        auto imgPath = (get_data_path() + input_file);
        FormatReader::ReaderPtr reader(imgPath.c_str());
        if (reader.get() == nullptr) {
            THROW_IE_EXCEPTION << "Can't read image file: " << imgPath;
        }

        auto image_data = reader.get()->getData().get();
        memcpy(input_buffer, image_data, reader->size());
    }

    void Infer()
    {
        StatusCode st;
        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    void GetDetectedFaces(std::vector<Face>& detected_faces)
    {
        StatusCode st;
        Blob::Ptr output;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("custom_fd", output, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        const int output_elem = 16;
        const int max_num_faces = 32;

        Face detected_face;
        auto *outBlob_data = output->buffer().as<PrecisionTrait<Precision::FP16>::value_type *>();
        for (int i = 0; i < max_num_faces; ++i)
        {
            auto face_left     = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 0]);
            auto face_top      = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 1]);
            auto face_right    = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 2]);
            auto face_bottom   = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 3]);
            auto confidence    = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 4]);
            auto tracking_id   = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 5]);
            auto left_eye_x    = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 6]);
            auto left_eye_y    = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 7]);
            auto right_eye_x   = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 8]);
            auto right_eye_y   = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 9]);
            auto nose_x        = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 10]);
            auto nose_y        = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 11]);
            auto mouth_left_x  = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 12]);
            auto mouth_left_y  = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 13]);
            auto mouth_right_x = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 14]);
            auto mouth_right_y = PrecisionUtils::f16tof32(outBlob_data[i * output_elem + 15]);

            if (confidence > 0) {
                detected_face.bbox = Rect(static_cast<int>(face_left + 0.5),
                                          static_cast<int>(face_top + 0.5),
                                          static_cast<int>(face_right - face_left + 0.5),
                                          static_cast<int>(face_bottom - face_top + 0.5));
                detected_face.left_eye  = Pos(left_eye_x, left_eye_y);
                detected_face.right_eye = Pos(right_eye_x, right_eye_y);
                detected_face.nose = Pos(nose_x, nose_y);
                detected_face.mouth_left_end = Pos(mouth_left_x, mouth_left_y);
                detected_face.mouth_right_end = Pos(mouth_right_x, mouth_right_y);
                detected_faces.push_back(detected_face);
                printf("Face detector: face at %d %d %d %d\n", (int)face_left, (int)face_top, (int)face_right, (int)face_bottom);
            }
        }
    }

};


TEST_P(myriadMTCNNTests_nightly, MTCNN)
{
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const MTCNNInputData input_data = std::tr1::get<0>(GetParam());
    const MTCNNTestParams test_params = std::tr1::get<1>(GetParam());
    const MTCNNInputShiftScales test_scale_shifts = std::tr1::get<2>(GetParam());
    const MTCNNResources test_resource_constraints = std::tr1::get<3>(GetParam());
    const SizeVector dims = input_data.dims;

    std::string model = GenModel(test_params, dims);
    auto weights_ptr = GenWeights(test_scale_shifts.scale, test_scale_shifts.b_shift, test_scale_shifts.g_shift, test_scale_shifts.r_shift);

    ASSERT_NO_FATAL_FAILURE(InitNetwork(model, weights_ptr, test_resource_constraints.resources));
    ASSERT_NO_FATAL_FAILURE(SetInput(input_data.input_path));
    ASSERT_NO_FATAL_FAILURE(Infer());

    std::vector<Face> detected_faces;
    ASSERT_NO_FATAL_FAILURE(GetDetectedFaces(detected_faces));

    std::vector<Face> gt_faces = input_data.gt_values;
    ASSERT_EQ(true, CompareResult(gt_faces, detected_faces));
}

static std::vector<mtcnn_input_scale_shifts> s_AVAFD_TestInputScaleShifts = {
        {{1.0f, -88.f, -97.f, -127.f}},
};

static std::vector<mtcnn_resouce_constraints> s_AVAFD_TestResourceConstraints = {
        {{-1}}, {{5}}, {{8}}
};

static std::vector<mtcnn_input_data> s_AVAFD_TestInputData = {
        {{{{1, 3, 1080, 1920}}, "/1920x1080/1920x1080.bmp",      AVA_gt_1920_1080_faces}},
        {{{{1, 3,  720, 1280}}, "/1280x720/1280x720_9faces.bmp", AVA_gt_1280_720_faces }},
        {{{{1, 3,  270, 480 }}, "/480x270/480x270.bmp",          AVA_gt_270_480_faces  }}
};

static std::vector<mtcnn_test_params> s_AVAFD_TestParam = {
        {{AVA_FD, "/ava-fd/12net_960x540/myriad_12net.xml", "/ava-fd/24net_24x24_conv/myriad_24net_conv.xml", "/ava-fd/48net_48x48_conv/myriad_48net_conv.xml", "240x135,120x67,60x33,30x16"}}
};

INSTANTIATE_TEST_CASE_P(AVAFD_MTCNN_TEST, myriadMTCNNTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_AVAFD_TestInputData),
            ::testing::ValuesIn(s_AVAFD_TestParam),
            ::testing::ValuesIn(s_AVAFD_TestInputScaleShifts),
            ::testing::ValuesIn(s_AVAFD_TestResourceConstraints))
);

static std::vector<mtcnn_test_params> s_AVAFD_zbatch_TestParam = {
        {{AVA_FD, "/ava-fd/12net_960x540/myriad_12net.xml", "/ava-fd/24net_24x24_conv_2batch_zdir/myriad_24net_conv_for_nce_2batch_zdir.xml", "/ava-fd/48net_48x48_conv/myriad_48net_conv.xml", "240x135,120x67,60x33,30x16"}}
};

INSTANTIATE_TEST_CASE_P(AVAFD_MTCNN_BATCH_TEST, myriadMTCNNTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_AVAFD_TestInputData),
            ::testing::ValuesIn(s_AVAFD_zbatch_TestParam),
            ::testing::ValuesIn(s_AVAFD_TestInputScaleShifts),
            ::testing::ValuesIn(s_AVAFD_TestResourceConstraints))
);

static std::vector<mtcnn_input_scale_shifts> s_MTCNN_TestInputScaleShifts = {
        {{2.0f, -127.f, -127.f, -127.f}},
};

static std::vector<mtcnn_resouce_constraints> s_MTCNN_TestResources = {
        {{-1}}, {{5}}, {{8}}
};

static std::vector<mtcnn_input_data> s_MTCNN_TestInputData = {
        {{{{1, 3, 270, 480}}, "/480x270/480x270.bmp", PUBLIC_gt_270_480_faces}},
};

static std::vector<mtcnn_test_params> s_MTCNN_TestParam = {
        {{PUBLIC_MTCNN, "/mtcnn/det1_288x162/det1.xml", "/mtcnn/det2_24x24_conv/det2_conv.xml", "/mtcnn/det3_48x48_conv/det3_conv.xml", "288x162,205x115,145x82,103x58,73x41,52x30,37x21,26x15"}}
};

INSTANTIATE_TEST_CASE_P(PUBLIC_MTCNN_TEST, myriadMTCNNTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_MTCNN_TestInputData),
            ::testing::ValuesIn(s_MTCNN_TestParam),
            ::testing::ValuesIn(s_MTCNN_TestInputScaleShifts),
            ::testing::ValuesIn(s_MTCNN_TestResources))
);

static std::vector<mtcnn_test_params> s_MTCNN_zbatch_TestParam = {
        {{PUBLIC_MTCNN, "/mtcnn/det1_288x162/det1.xml", "/mtcnn/det2_24x24_conv_2batch_zdir/det2_conv_2batch_zdir.xml", "/mtcnn/det3_48x48_conv/det3_conv.xml", "288x162,205x115,145x82,103x58,73x41,52x30,37x21,26x15"}}
};

INSTANTIATE_TEST_CASE_P(PUBLIC_MTCNN_BATCH_TEST, myriadMTCNNTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_MTCNN_TestInputData),
            ::testing::ValuesIn(s_MTCNN_zbatch_TestParam),
            ::testing::ValuesIn(s_MTCNN_TestInputScaleShifts),
            ::testing::ValuesIn(s_MTCNN_TestResources))
);

}
