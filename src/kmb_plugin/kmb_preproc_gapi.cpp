// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__arm__) || defined(__aarch64__)
#include "kmb_preproc_gapi.hpp"

#include <ie_blob.h>
#include <ie_compound_blob.h>

#include <memory>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi_m2i/preproc.hpp>
#include <opencv2/gapi_sipp/sippinitinfo.hpp>
#include <utility>
#include <vector>

#include "debug.h"
#include "kmb_preproc_gapi_kernels.hpp"
#include "kmb_preproc_gapi_kernels_sipp.hpp"

// clang-format off
namespace InferenceEngine {
namespace KmbPreproc {

namespace detail {

struct BlobDesc {
    Precision prec;
    Layout layout;
    SizeVector sz_v;
    ColorFormat fmt;
    bool operator!=(const BlobDesc& other) const {
        return other.prec != prec || other.layout != layout || other.sz_v != sz_v || other.fmt != fmt;
    }
};

struct CallDesc {
    BlobDesc in_desc;
    BlobDesc out_desc;
    ResizeAlgorithm alg;
};

} // namespace detail

class PreprocEngine::Priv {
public:
    virtual ~Priv() = default;

    virtual void go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                    const ResizeAlgorithm& algorithm,
                    ColorFormat in_fmt, ColorFormat out_fmt) = 0;
};

class PrivSIPP final: public PreprocEngine::Priv {
    std::unique_ptr<cv::GComputation> _comp = nullptr;
    unsigned int _shaveFirst;
    unsigned int _shaveLast;
    unsigned int _lpi;

public:
    template<typename T> using Opt = cv::util::optional<T>;

    Opt<detail::CallDesc> _lastCall;
    cv::GCompiled _lastComp;

    enum class Update { REBUILD, RESHAPE, NOTHING };
    Update needUpdate(const detail::CallDesc &newCall) const;

    void updateGraph(const detail::CallDesc& callDesc,
                     const TensorDesc& in_desc_ie,
                     const TensorDesc& out_desc_ie,
                     const Blob::Ptr& inBlob,
                     const std::vector<cv::gapi::own::Mat>& input_plane_mats,
                     Update update);

    void executeGraph(const std::vector<cv::gapi::own::Mat>& input_plane_mats,
                            std::vector<cv::gapi::own::Mat>& output_plane_mats);

    template<typename BlobTypePtr>
    void preprocessBlob(const BlobTypePtr &inBlob, MemoryBlob::Ptr &outBlob,
                        ResizeAlgorithm algorithm,
                        ColorFormat in_fmt, ColorFormat out_fmt);

    PrivSIPP(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi)
        : _shaveFirst(shaveFirst)
        , _shaveLast(shaveLast)
        , _lpi(lpi) {
    }

    void go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
            const ResizeAlgorithm& algorithm,
            ColorFormat in_fmt, ColorFormat out_fmt) override;
};

class PrivM2I final: public PreprocEngine::Priv {
    std::unique_ptr<cv::GComputation> _comp = nullptr;

public:
    void go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
            const ResizeAlgorithm& algorithm,
            ColorFormat in_fmt, ColorFormat out_fmt) override;
};

namespace {
namespace G {
    struct Strides {int N; int C; int H; int W;};
    struct Dims    {int N; int C; int H; int W;};
    struct Desc    {Dims d; Strides s;};

    void fix_strides_nhwc(const Dims &d, Strides &s) {
        if (s.W > d.C) {
            s.C = 1;
            s.W = s.C*d.C;
            s.H = s.W*d.W;
            s.N = s.H*d.H;
        }
    }

    Desc decompose(const TensorDesc& ie_desc) {
        const auto& ie_blk_desc = ie_desc.getBlockingDesc();
        const auto& ie_dims     = ie_desc.getDims();
        const auto& ie_strides  = ie_blk_desc.getStrides();
        const bool  nhwc_layout = ie_desc.getLayout() == NHWC;

        Dims d = {
            static_cast<int>(ie_dims[0]),
            static_cast<int>(ie_dims[1]),
            static_cast<int>(ie_dims[2]),
            static_cast<int>(ie_dims[3])
        };

        Strides s = {
            static_cast<int>(ie_strides[0]),
            static_cast<int>(nhwc_layout ? ie_strides[3] : ie_strides[1]),
            static_cast<int>(nhwc_layout ? ie_strides[1] : ie_strides[2]),
            static_cast<int>(nhwc_layout ? ie_strides[2] : ie_strides[3]),
        };

        if (nhwc_layout) fix_strides_nhwc(d, s);

        return Desc{d, s};
    }

    Desc decompose(const Blob::Ptr& blob) {
        return decompose(blob->getTensorDesc());
    }
}  // namespace G

inline int get_cv_depth(const TensorDesc &ie_desc) {
    switch (ie_desc.getPrecision()) {
    case Precision::U8:   return CV_8U;
    default: THROW_IE_EXCEPTION << "Unsupported data type";
    }
}

cv::gapi::own::Size getFullImageSize(const Blob::Ptr& blob) {
    const auto desc = blob->getTensorDesc();
    auto strides = desc.getBlockingDesc().getStrides();
    cv::gapi::own::Size sz;

    if (desc.getLayout() == Layout::NHWC) {
        int w = strides[1] / strides[2];
        int h = strides[0] / strides[1];
        sz = {w, h};
    } else if (desc.getLayout() == Layout::NCHW) { // FIXME: need to verify
        int w = strides[2] / strides[3];
        int h = strides[0] / strides[2];
        sz = {w, h};
    } else {
        THROW_IE_EXCEPTION << "Unsupported layout";
    }

    return sz;
}

std::vector<cv::gapi::own::Mat> bind_to_blob(const Blob::Ptr& blob) {
    const auto& ie_desc     = blob->getTensorDesc();
    const auto& ie_desc_blk = ie_desc.getBlockingDesc();
    const auto     desc     = G::decompose(blob);
    const auto cv_depth     = get_cv_depth(ie_desc);
    const auto stride       = desc.s.H*blob->element_size();
    const auto planeSize    = cv::gapi::own::Size(desc.d.W, desc.d.H);

    uint8_t* blob_ptr = static_cast<uint8_t*>(blob->buffer());
    if (blob_ptr == nullptr) {
        THROW_IE_EXCEPTION << "Blob buffer is nullptr";
    }
    blob_ptr += blob->element_size()*ie_desc_blk.getOffsetPadding();

    std::vector<cv::gapi::own::Mat> result;

    if (ie_desc.getLayout() == Layout::NHWC) {
        result = {{planeSize.height, planeSize.width, CV_MAKETYPE(cv_depth, desc.d.C),
                  blob_ptr, stride}};
    } else {  // NCHW
        if (desc.d.C <= 0) {
            THROW_IE_EXCEPTION << "Invalid number of channels in blob tensor descriptor, "
                                  "expected >0, actual: " << desc.d.C;
        }

        const auto planeType = CV_MAKETYPE(cv_depth, 1);
        result = {{planeSize.height*desc.d.C, planeSize.width, planeType,
                  blob_ptr, stride}};
    }

    return result;
}

cv::gapi::own::Mat bind_to_blob(const NV12Blob::Ptr& blob) {
    // This is a special case for M2I & NV12.
    // FIXME: M2I has a single input only!
    // Even for NV12. It means the whole NV12 buffer
    // is passed in as a plain continious memory region.
    // What we need to validate here is that the uv data pointer
    // really comes right after the y data ends:
    const auto& y_blob  = blob->y();
    const auto& uv_blob = blob->uv();
    auto input_y   = bind_to_blob(y_blob).back();
    auto input_uv  = bind_to_blob(uv_blob).back();
    if (input_y.data == nullptr) {
        THROW_IE_EXCEPTION << "input_y.data is nullptr";
    }
    if (input_uv.data == nullptr) {
        THROW_IE_EXCEPTION << "input_uv.data is nullptr";
    }
    if (input_uv.data != input_y.data + input_y.rows*input_y.step) {
        THROW_IE_EXCEPTION << "Input NV12 memory is not continious";
    }

    // Extract the memory description based on Y plane only
    const auto& ie_desc     = y_blob->getTensorDesc();
    IE_ASSERT(ie_desc.getPrecision() == Precision::U8);
    const auto& ie_desc_blk = ie_desc.getBlockingDesc();
    const auto     desc     = G::decompose(y_blob);
    IE_ASSERT(desc.d.H  % 2 == 0);
    const auto stride       = desc.s.H*blob->element_size();
    const auto size         = cv::gapi::own::Size(desc.d.W, (desc.d.H/2)*3);

    IE_ASSERT(ie_desc_blk.getOffsetPadding() == 0);
    return {size.height, size.width, CV_8UC1, input_y.data, stride};
}

// validate input/output ColorFormat-related parameters
void validateColorFormats(const G::Desc &in_desc,
                          const G::Desc &out_desc,
                          Layout in_layout,
                          Layout out_layout,
                          ColorFormat input_color_format,
                          ColorFormat output_color_format) {
    const auto verify_desc = [] (const G::Desc& desc, ColorFormat fmt, const std::string& desc_prefix) {
        const auto throw_invalid_number_of_channels = [&](){
            THROW_IE_EXCEPTION << desc_prefix << " tensor descriptor "
                               << "has invalid number of channels "
                               << desc.d.C << " for " << fmt
                               << "color format";
        };
        switch (fmt) {
            case ColorFormat::NV12: {
                if (desc.d.C != 2) throw_invalid_number_of_channels();
                break;
            }
            case ColorFormat::RGB:
            case ColorFormat::BGR: {
                if (desc.d.C != 3) throw_invalid_number_of_channels();
                break;
            }
            case ColorFormat::RGBX:
            case ColorFormat::BGRX: {
                if (desc.d.C != 4) throw_invalid_number_of_channels();
                break;
            }

            default: break;
        }
    };

    const auto verify_layout = [] (Layout layout, const std::string& layout_prefix) {
        if (layout != NHWC && layout != NCHW) {
            THROW_IE_EXCEPTION << layout_prefix << " layout " << layout
                               << " is not supported by pre-processing [by G-API]";
        }
    };

    // verify inputs/outputs and throw on error

    if (output_color_format == ColorFormat::RAW) {
        THROW_IE_EXCEPTION << "Network's expected color format is unspecified";
    }

    if (output_color_format == ColorFormat::NV12 || output_color_format == ColorFormat::I420) {
        THROW_IE_EXCEPTION << "NV12/I420 network's color format is not supported [by G-API]";
    }

    verify_layout(in_layout,  "Input blob");
    verify_layout(out_layout, "Network's blob");

    if (input_color_format == ColorFormat::RAW) {
        // verify input and output have the same number of channels
        if (in_desc.d.C != out_desc.d.C) {
            THROW_IE_EXCEPTION << "Input and network expected blobs have different number of "
                               << "channels: expected " << out_desc.d.C << " channels but provided "
                               << in_desc.d.C << " channels";
        }
        return;
    }

    // planar 4-channel input is not supported, user can easily pass 3 channels instead of 4
    if (in_layout == NCHW
        && (input_color_format == ColorFormat::RGBX || input_color_format == ColorFormat::BGRX)) {
        THROW_IE_EXCEPTION << "Input blob with NCHW layout and BGRX/RGBX color format is "
                           << "explicitly not supported, use NCHW + BGR/RGB color format "
                           << "instead (3 image planes instead of 4)";
    }

    // verify input and output against their corresponding color format
    verify_desc(in_desc, input_color_format, "Input blob");
    verify_desc(out_desc, output_color_format, "Network's blob");
}

bool has_zeros(const SizeVector& vec) {
    return std::any_of(vec.cbegin(), vec.cend(), [] (size_t e) { return e == 0; });
}

void validateTensorDesc(const TensorDesc& desc) {
    auto supports_layout = [](Layout l) { return l == Layout::NCHW || l == Layout::NHWC; };
    const auto layout = desc.getLayout();
    const auto& dims = desc.getDims();
    if (!supports_layout(layout)
        || dims.size() != 4
        || desc.getBlockingDesc().getStrides().size() != 4) {
        THROW_IE_EXCEPTION << "Preprocess support NCHW/NHWC only";
    }
    if (has_zeros(dims)) {
        THROW_IE_EXCEPTION << "Invalid input data dimensions: "
                           << details::dumpVec(dims);
    }
}

void validateBlob(const MemoryBlob::Ptr &) {}

void validateBlob(const NV12Blob::Ptr &inBlob) {
    const auto& y_blob = inBlob->y();
    const auto& uv_blob = inBlob->uv();
    if (!y_blob || !uv_blob) {
        THROW_IE_EXCEPTION << "Invalid underlying blobs in NV12Blob";
    }

    validateTensorDesc(uv_blob->getTensorDesc());
}

const std::pair<const TensorDesc&, Layout> getTensorDescAndLayout(const MemoryBlob::Ptr &blob) {
    const auto& desc =  blob->getTensorDesc();
    return {desc, desc.getLayout()};
}

// use Y plane tensor descriptor's dims for tracking if update is needed. Y and U V planes are
// strictly bound to each other: if one is changed, the other must be changed as well. precision
// is always U8 and layout is always planar (NCHW)
// FIXME: Not sure about the layout
const std::pair<const TensorDesc&, Layout> getTensorDescAndLayout(const NV12Blob::Ptr &blob) {
    return {blob->y()->getTensorDesc(), Layout::NCHW};
}

G::Desc getGDesc(G::Desc in_desc_y, const NV12Blob::Ptr &) {
    auto nv12_desc = G::Desc{};
    nv12_desc.d = in_desc_y.d;
    nv12_desc.d.C = 2;

    return nv12_desc;
}

G::Desc getGDesc(G::Desc in_desc_y, const MemoryBlob::Ptr &) {
    return in_desc_y;
}

// FIXME: need to hide somewhere?
class PreprocGraphBuilder {
public:
    PreprocGraphBuilder(G::Desc out_desc_,
                        Layout in_layout_,
                        Layout out_layout_,
                        ResizeAlgorithm algorithm_,
                        ColorFormat input_color_format_,
                        ColorFormat output_color_format_) :
                             input_color_format(input_color_format_),
                             output_color_format(output_color_format_),
                             out_desc(out_desc_),
                             in_layout(in_layout_),
                             out_layout(out_layout_),
                             algorithm(algorithm_) {}

    cv::GComputation build() {
        // First, do color convertions:
        // 1) NV12 -> RGB/BGR
        // 2) RGBX/BGRX -> RGB/BGR
        // 3) RGB/BGR -> RGB/BGR
        // 4) RAW - no conversions
        // Then resize with optional merge
        switch (input_color_format) {
            case ColorFormat::NV12: return NV12Preproc();
            case ColorFormat::RGBX: return RGBXPreproc();
            case ColorFormat::BGRX: return BGRXPreproc();
            case ColorFormat::RGB:  return RGBPreproc();
            case ColorFormat::BGR:  return BGRPreproc();
            case ColorFormat::RAW:  return RAWPreproc();
            default : THROW_IE_EXCEPTION << "Unsupported input color format";
        }
    }

private:
    cv::GMatP inp, converted;
    cv::GMat in, y, uv;

    cv::GMatP resize(cv::GMatP src) {
        cv::GMatP dst;
        if (algorithm != NO_RESIZE) {
            const int interp_type = [](const ResizeAlgorithm &ar) {
                switch (ar) {
                case RESIZE_AREA:     return cv::INTER_AREA;
                case RESIZE_BILINEAR: return cv::INTER_LINEAR;
                default: THROW_IE_EXCEPTION << "Unsupported resize operation";
                }
            } (algorithm);
            const auto scale_sz  = cv::gapi::own::Size(out_desc.d.W, out_desc.d.H);
            // FIXME: Not sure if it will work with 1 channel RAW image (NCHW/NHWC issue)
            // dst = gapi::scalePlanes(src, scale_sz, interp_type);
            dst = gapi::resizeP(src, scale_sz, interp_type);
        } else {
            dst = src;
        }
     return dst;
    }

    cv::GMat merge(cv::GMatP src, int chan) {
        switch (chan) {
            case 2: return gapi::merge2(src);
            case 3: return gapi::merge3p(src);
            case 4: return gapi::merge4(src);
            default: THROW_IE_EXCEPTION << "Unsupported number of channels";
        }
    }

    cv::GMat resizeAndMerge(cv::GMatP src) {
        auto dst = resize(src);
        return merge(dst, out_desc.d.C);
    }

    cv::GComputation resizeWithOptionalMerge(cv::GProtoInputArgs in, cv::GMatP converted) {
        if (out_layout == NHWC) {
            return cv::GComputation(std::move(in),
                    cv::GOut(resizeAndMerge(converted)));
        } else { // NCHW
            return cv::GComputation(std::move(in),
                    cv::GOut(resize(converted)));
        }
    }

    cv::GComputation NV12Preproc() {
        if (output_color_format == ColorFormat::RGB) {
            converted = gapi::NV12toRGBp(y, uv);
        } else { // BGR
            converted = gapi::NV12toBGRp(y, uv);
        }
        return resizeWithOptionalMerge(cv::GIn(y, uv), converted);
    }

    cv::GComputation RGBXPreproc() {
        if (output_color_format == ColorFormat::RGB) {
            converted = gapi::drop4(inp);
        } else { // BGR
            converted = gapi::swapChan(gapi::drop4(inp));
        }
        return resizeWithOptionalMerge(cv::GIn(inp), converted);
    }

    cv::GComputation BGRXPreproc() {
        if (output_color_format == ColorFormat::BGR) {
            converted = gapi::drop4(inp);
        } else { // RGB
            converted = gapi::swapChan(gapi::drop4(inp));
        }
        return resizeWithOptionalMerge(cv::GIn(inp), converted);
    }

    cv::GComputation RGBPreproc() {
        if (in_layout == NHWC) {
            inp = gapi::interleaved2planar(in);
            converted = inp;
            if (output_color_format == ColorFormat::BGR) {
                converted = gapi::swapChan(inp);
            }
            return resizeWithOptionalMerge(cv::GIn(in), converted);
        } else { // NCHW
            converted = inp;
            if (output_color_format == ColorFormat::BGR) {
                converted = gapi::swapChan(inp);
            }
            return resizeWithOptionalMerge(cv::GIn(inp), converted);
        }
    }

    cv::GComputation BGRPreproc() {
        if (in_layout == NHWC) {
            inp = gapi::interleaved2planar(in);
            converted = inp;
            if (output_color_format == ColorFormat::RGB) {
                converted = gapi::swapChan(inp);
            }
            return resizeWithOptionalMerge(cv::GIn(in), converted);
        } else { // NCHW
            converted = inp;
            if (output_color_format == ColorFormat::RGB) {
                converted = gapi::swapChan(inp);
            }
            return resizeWithOptionalMerge(cv::GIn(inp), converted);
        }
    }

    cv::GComputation RAWPreproc() {
        if (in_layout == NHWC) {
            converted = gapi::interleaved2planar(in);
            return resizeWithOptionalMerge(cv::GIn(in), converted);
        } else { // NCHW
            return resizeWithOptionalMerge(cv::GIn(converted), converted);
        }
    }

    ColorFormat input_color_format, output_color_format;
    G::Desc out_desc;
    Layout in_layout, out_layout;
    ResizeAlgorithm algorithm;
};

cv::GComputation getPreprocGraph(const G::Desc &in_desc,
                                 const G::Desc &out_desc,
                                 Layout in_layout,
                                 Layout out_layout,
                                 ResizeAlgorithm algorithm,
                                 ColorFormat input_color_format,
                                 ColorFormat output_color_format,
                                 int precision) {
    // perform basic validation to ensure our assumptions about input and output are correct
    validateColorFormats(in_desc, out_desc, in_layout, out_layout, input_color_format,
                         output_color_format);

    // The only supported precision for now
    IE_ASSERT(precision == CV_8U);

    PreprocGraphBuilder ppGraph(out_desc,
                                in_layout,
                                out_layout,
                                algorithm,
                                input_color_format,
                                output_color_format);

    return ppGraph.build();
}

} // anonymous namespace

PrivSIPP::Update PrivSIPP::needUpdate(const detail::CallDesc &newCallOrig) const {
    // Given our knowledge about G-API, full graph rebuild is required
    // if and only if:
    // 0. This is the first call ever
    // 1. precision has changed (affects kernel versions)
    // 2. layout has changed (affects graph topology)
    // 3. algorithm has changed (affects kernel version)
    // 4. dimensions have changed from downscale to upscale or vice-versa if interpolation is AREA
    // 5. color format has changed (affects graph topology)
    if (!_lastCall) {
        return Update::REBUILD;
    }

    detail::BlobDesc last_in = _lastCall->in_desc;
    detail::BlobDesc last_out = _lastCall->out_desc;
    ResizeAlgorithm last_algo = _lastCall->alg;

    detail::CallDesc newCall = newCallOrig;
    detail::BlobDesc new_in = newCall.in_desc;
    detail::BlobDesc new_out = newCall.out_desc;
    ResizeAlgorithm new_algo = newCall.alg;

    // Declare two empty vectors per each call
    SizeVector last_in_size;
    SizeVector last_out_size;
    SizeVector new_in_size;
    SizeVector new_out_size;

    // Now swap it with in/out descriptor vectors
    // Now last_in/last_out would contain everything but sizes
    last_in_size.swap(last_in.sz_v);
    last_out_size.swap(last_out.sz_v);
    new_in_size.swap(new_in.sz_v);
    new_out_size.swap(new_out.sz_v);

    // If anything (except input sizes) changes, rebuild is required
    if (last_in != new_in || last_out != new_out || last_algo != new_algo) {
        return Update::REBUILD;
    }

    // If output sizes change, graph should be regenerated (resize
    // ratio is taken from parameters)
    if (last_out_size != new_out_size) {
        return Update::REBUILD;
    }

    // If interpolation is AREA and sizes change upscale/downscale
    // mode, rebuild is required
    if (last_algo == RESIZE_AREA) {
        // 0123 == NCHW
        const auto is_upscale = [](const SizeVector &in, const SizeVector &out) -> bool {
            return in[2] < out[2] || in[3] < out[3];
        };
        const bool old_upscale = is_upscale(last_in_size, last_out_size);
        const bool new_upscale = is_upscale(new_in_size, new_out_size);
        if (old_upscale != new_upscale) {
            return Update::REBUILD;
        }
    }

    // If only input sizes changes (considering the above exception),
    // reshape is enough
    if (last_in_size != new_in_size) {
        return Update::RESHAPE;
    }

    return Update::NOTHING;
}

void PrivSIPP::updateGraph(const detail::CallDesc& callDesc,
                           const TensorDesc& in_desc_ie,
                           const TensorDesc& out_desc_ie,
                           const Blob::Ptr& inBlob,
                           const std::vector<cv::gapi::own::Mat>& input_plane_mats,
                           Update update) {
    _lastCall = cv::util::make_optional(std::move(callDesc));
    detail::BlobDesc new_in  = _lastCall.value().in_desc;
    detail::BlobDesc new_out = _lastCall.value().out_desc;
    auto new_algo = _lastCall.value().alg;

    Layout in_layout = new_in.layout;
    ColorFormat in_fmt = new_in.fmt;

    Layout out_layout = new_out.layout;
    ColorFormat out_fmt = new_out.fmt;

    cv::GSIPPMaxFrameSizes max_sizes;
    if (in_fmt == ColorFormat::NV12) {
        IE_ASSERT(input_plane_mats.size() == 2);
        max_sizes.sizes.emplace_back(getFullImageSize(as<NV12Blob>(inBlob)->y()));
        max_sizes.sizes.emplace_back(getFullImageSize(as<NV12Blob>(inBlob)->uv()));
    } else {
        IE_ASSERT(input_plane_mats.size() == 1);
        max_sizes.sizes.emplace_back(getFullImageSize(as<MemoryBlob>(inBlob)));
    }

    auto args = cv::compile_args(InferenceEngine::gapi::preproc::sipp::kernels(),
                                 cv::GSIPPBackendInitInfo {_shaveFirst, _shaveLast, _lpi},
                                 max_sizes);

    if (Update::REBUILD == update) {
        //  rebuild the graph
        const G::Desc
            in_desc =  G::decompose(in_desc_ie),
            out_desc = G::decompose(out_desc_ie);

        // FIXME: what is a correct G::Desc to be passed for NV12 case?
        G::Desc custom_desc;
        if (in_fmt == ColorFormat::NV12) {
            custom_desc = getGDesc(in_desc, as<NV12Blob>(inBlob));
        } else {
            custom_desc = getGDesc(in_desc, as<MemoryBlob>(inBlob));
        }
        auto ppComputaton = cv::util::make_optional(
            getPreprocGraph(custom_desc,
                            out_desc,
                            in_layout,
                            out_layout,
                            new_algo,
                            in_fmt,
                            out_fmt,
                            get_cv_depth(in_desc_ie)));

        auto& computation = ppComputaton.value();
        _lastComp = computation.compile(descrs_of(input_plane_mats), std::move(args));
    } else {
        IE_ASSERT(_lastComp);
        _lastComp.reshape(descrs_of(input_plane_mats), std::move(args));
    }
}

void PrivSIPP::executeGraph(const std::vector<cv::gapi::own::Mat>& input_plane_mats,
                                  std::vector<cv::gapi::own::Mat>& output_plane_mats) {
    // FIXME:: not sure about "planarity" here
    cv::GRunArgs call_ins;
    cv::GRunArgsP call_outs;
    for (const auto & m : input_plane_mats) { call_ins.emplace_back(m); }
    for (auto & m : output_plane_mats) { call_outs.emplace_back(&m); }

    _lastComp(std::move(call_ins), std::move(call_outs));
}

namespace {
std::vector<cv::gapi::own::Mat> input_from_blob(const MemoryBlob::Ptr& inBlob) {
    return bind_to_blob(inBlob);
}

std::vector<cv::gapi::own::Mat> input_from_blob(const NV12Blob::Ptr& inBlob) {
    std::vector<cv::gapi::own::Mat> input_plane_mats;
    auto input_y  = bind_to_blob(inBlob->y());
    auto input_uv = bind_to_blob(inBlob->uv());
    input_plane_mats.push_back(input_y.back());
    input_plane_mats.push_back(input_uv.back());

    return input_plane_mats;
}
} // anonymous namespace

template<typename BlobTypePtr>
void PrivSIPP::preprocessBlob(const BlobTypePtr &inBlob, MemoryBlob::Ptr &outBlob,
                              ResizeAlgorithm algorithm,
                              ColorFormat in_fmt, ColorFormat out_fmt) {
    validateBlob(inBlob);

    auto desc_and_layout = getTensorDescAndLayout(inBlob);

    const auto& in_desc_ie = desc_and_layout.first;
    const auto  in_layout  = desc_and_layout.second;

    const auto& out_desc_ie = outBlob->getTensorDesc();
    const auto  out_layout = out_desc_ie.getLayout();

    validateTensorDesc(in_desc_ie);
    validateTensorDesc(out_desc_ie);

    detail::CallDesc thisCall = detail::CallDesc{ detail::BlobDesc{ in_desc_ie.getPrecision(),
                                                                    in_layout,
                                                                    in_desc_ie.getDims(),
                                                                    in_fmt },
                                                  detail::BlobDesc{ out_desc_ie.getPrecision(),
                                                                    out_layout,
                                                                    out_desc_ie.getDims(),
                                                                    out_fmt },
                                                  algorithm };

    const Update update = needUpdate(thisCall);

    std::vector<cv::gapi::own::Mat> input_plane_mats = input_from_blob(inBlob);
    auto output_plane_mats = bind_to_blob(outBlob);

    if (update != Update::NOTHING) {
        updateGraph(thisCall, in_desc_ie, out_desc_ie, inBlob, input_plane_mats,
                    update);
    }

    executeGraph(input_plane_mats, output_plane_mats);
}

void PrivSIPP::go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                  const ResizeAlgorithm& algorithm,
                  ColorFormat in_fmt, ColorFormat out_fmt) {
    // The only supported configuration for now
    IE_ASSERT(in_fmt == ColorFormat::NV12);
    IE_ASSERT(algorithm == RESIZE_BILINEAR);
    IE_ASSERT(out_fmt == ColorFormat::RGB || out_fmt == ColorFormat::BGR);

    // output is always a memory blob
    auto outMemoryBlob = as<MemoryBlob>(outBlob);
    if (!outMemoryBlob) {
        THROW_IE_EXCEPTION  << "Unsupported network's input blob type: expected MemoryBlob";
    }

    // If input color format is not NV12 (which is a future feature), a MemoryBlob is expected.
    // Otherwise, NV12Blob is expected.
    switch (in_fmt) {
    case ColorFormat::NV12: {
        auto inNV12Blob = as<NV12Blob>(inBlob);
        if (!inNV12Blob) {
            THROW_IE_EXCEPTION  << "Unsupported input blob for color format " << in_fmt
                                << ": expected NV12Blob";
        }
        preprocessBlob(inNV12Blob, outMemoryBlob, algorithm, in_fmt, out_fmt);
        break;
    }
    default:
        auto inMemoryBlob = as<MemoryBlob>(inBlob);
        if (!inMemoryBlob) {
            THROW_IE_EXCEPTION  << "Unsupported input blob for color format " << in_fmt
                                << ": expected MemoryBlob";
        }
        preprocessBlob(inMemoryBlob, outMemoryBlob, algorithm, in_fmt, out_fmt);
        break;
    }
}

void PrivM2I::go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                 const ResizeAlgorithm& algorithm,
                 ColorFormat in_fmt, ColorFormat out_fmt) {
    // NB.: Still follow the same constraints as with SIPP
    IE_ASSERT(algorithm == RESIZE_BILINEAR);
    IE_ASSERT(in_fmt == NV12);
    IE_ASSERT(out_fmt == ColorFormat::RGB || out_fmt == ColorFormat::BGR);

    if (out_fmt == ColorFormat::RGB) {
        THROW_IE_EXCEPTION << "M2I PP: RGB output color format is not supported";
    }

    auto inNV12Blob = as<NV12Blob>(inBlob);
    IE_ASSERT(inNV12Blob != nullptr);

    auto input  = bind_to_blob(inNV12Blob);
    auto output = bind_to_blob(outBlob).back(); // fixme: single output???

    // FIXME: add batch??

    if (!_comp) {
        cv::GMat in;
        cv::GMat  out_i; // in the case of interleaved output
        cv::GMatP out_p; // in the case of planar output

        const cv::gapi::m2i::CSC csc_code = [out_fmt]() {
            switch (out_fmt) {
            case ColorFormat::RGB: return cv::gapi::m2i::CSC::NV12toRGB;
            case ColorFormat::BGR: return cv::gapi::m2i::CSC::NV12toBGR;
            default: THROW_IE_EXCEPTION << "M2I PP: Unsupported color space conversion";
            }
        }();

        cv::gapi::own::Size out_sz{output.cols, output.rows};
        if (outBlob->getTensorDesc().getLayout() == NCHW) {
            // planar output case
            out_sz.height /= 3; // see details in bind_to_blob()
            out_p = cv::gapi::M2Ip(in, csc_code, out_sz);
            _comp.reset(new cv::GComputation(GIn(in), cv::GOut(out_p)));
        } else {
            // interleaved output case
            out_i = cv::gapi::M2Ii(in, csc_code, out_sz);
            _comp.reset(new cv::GComputation(GIn(in), cv::GOut(out_i)));
        }
        IE_ASSERT(_comp != nullptr);
    }
    _comp->apply(cv::gin(input), cv::gout(output),
                 cv::compile_args(cv::gapi::preproc::m2i::kernels()));
}

PreprocEngine::PreprocEngine(unsigned int shaveFirst, unsigned int shaveLast,
                             unsigned int lpi, Path ppPath) {
    IE_ASSERT(ppPath == Path::SIPP || ppPath == Path::M2I);
    if (ppPath == Path::SIPP) {
        _priv.reset(new PrivSIPP(shaveFirst, shaveLast, lpi));
    } else if (ppPath == Path::M2I) {
        _priv.reset(new PrivM2I());
    } else {
        THROW_IE_EXCEPTION << "Error: unsupported preprocessing path with code "
                           << std::to_string(static_cast<int>(ppPath));
    }
}

PreprocEngine::~PreprocEngine() = default;

void PreprocEngine::preproc(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                            const ResizeAlgorithm& algorithm,
                            ColorFormat in_fmt, ColorFormat out_fmt) {
    return _priv->go(inBlob, outBlob, algorithm, in_fmt, out_fmt);
}

}  // namespace KmbPreproc
}  // namespace InferenceEngine
// clang-format on
#endif
