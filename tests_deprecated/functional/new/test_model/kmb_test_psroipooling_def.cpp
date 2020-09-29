//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "kmb_test_psroipooling_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/op/psroi_pooling.hpp>

namespace {

BlobVector refPSROIPooling(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    auto shape = layer->get_shape();
    IE_ASSERT(shape.size() == 4);

    SizeVector out_dims{shape[0], shape[1], shape[2], shape[3]};
    auto out_desc = TensorDesc(Precision::FP32, out_dims, Layout::NCHW);
    auto output = make_blob_with_precision(out_desc);
    output->allocate();

    const auto psroi = std::dynamic_pointer_cast<ngraph::op::PSROIPooling>(layer);

    const auto& in_desc = inputs[0]->getTensorDesc();
    const int channels  = in_desc.getDims()[1];
    const int height    = in_desc.getDims()[2];
    const int width     = in_desc.getDims()[3];

    IE_ASSERT(psroi->get_output_size() == 1u);
    const auto& output_shape = psroi->get_output_shape(0);
    const int nn = output_shape[0];
    const int nc = output_shape[1];
    const int nh = output_shape[2];
    const int nw = output_shape[3];

    float spatial_scale      = psroi->get_spatial_scale();
    const int pooled_height  = psroi->get_group_size();
    const int pooled_width   = psroi->get_group_size();
    const int group_size     = psroi->get_group_size();
    const auto& mode         = psroi->get_mode();
    IE_ASSERT(mode == "average");

    float* dst_data = output->buffer().as<float*>();
    const float* bottom_data_beginning = inputs[0]->buffer().as<float*>();
    const float* bottom_rois_beginning = inputs[1]->buffer().as<float*>();

    for (int n = 0; n < nn; ++n) {
        const float* bottom_rois = bottom_rois_beginning + n * 5;
        int roi_batch_ind = static_cast<int>(bottom_rois[0]);

        auto roi_start_w = static_cast<float>(round(bottom_rois[1])) * spatial_scale;
        auto roi_start_h = static_cast<float>(round(bottom_rois[2])) * spatial_scale;
        auto roi_end_w   = static_cast<float>(round(bottom_rois[3]) + 1.0f) * spatial_scale;
        auto roi_end_h   = static_cast<float>(round(bottom_rois[4]) + 1.0f) * spatial_scale;
        // Force too small ROIs to be 1x1
        auto roi_width  = std::max<float>(roi_end_w - roi_start_w, 0.1f);  // avoid 0
        auto roi_height = std::max<float>(roi_end_h - roi_start_h, 0.1f);

        for (int c = 0; c < nc; c++) {
            for (int h = 0; h < nh; h++) {
                for (int w = 0; w < nw; w++) {
                    size_t index = n*nc*nh*nw + c*nh*nw + h*nw + w;
                    dst_data[index] = 0.0f;

                    float bin_size_h = roi_height / static_cast<float>(pooled_height);
                    float bin_size_w = roi_width  / static_cast<float>(pooled_width);

                    int hstart = static_cast<int>(floor(static_cast<float>(h + 0) * bin_size_h + roi_start_h));
                    int hend = static_cast<int>(ceil(static_cast<float>(h + 1) * bin_size_h + roi_start_h));

                    hstart = std::min<int>(std::max<int>(hstart, 0), height);
                    hend = std::min<int>(std::max<int>(hend, 0), height);
                    int wstart = static_cast<int>(floor(static_cast<float>(w + 0) * bin_size_w + roi_start_w));
                    int wend = static_cast<int>(ceil(static_cast<float>(w + 1) * bin_size_w + roi_start_w));

                    wstart = std::min<int>(std::max<int>(wstart, 0), width);
                    wend = std::min<int>(std::max<int>(wend, 0), width);

                    float bin_area = static_cast<float>((hend - hstart) * (wend - wstart));
                    if (bin_area) {
                        int gc = (c * group_size + h) * group_size + w;
                        const float *bottom_data =
                                bottom_data_beginning + ((roi_batch_ind * channels + gc) * height * width);

                        float out_sum = 0.0f;
                        for (int hh = hstart; hh < hend; ++hh)
                            for (int ww = wstart; ww < wend; ++ww)
                                out_sum += bottom_data[hh * width + ww];

                        dst_data[index] = out_sum / bin_area;
                    }
                }
            }
        }
    }

    return {output};
};

}  // namespace

TestNetwork& PSROIPoolingLayerDef::build() {
    auto input  = net_.getPort(input_port_);
    auto coords = net_.getPort(coords_port_);

    auto psroi  = std::make_shared<ngraph::op::PSROIPooling>(input,
                                                             coords,
                                                             params_.output_dim_,
                                                             params_.group_size_,
                                                             params_.spatial_scale_,
                                                             params_.spatial_bin_x_,
                                                             params_.spatial_bin_y_,
                                                             params_.mode_);
    return net_.addLayer(name_, psroi, refPSROIPooling);
}
