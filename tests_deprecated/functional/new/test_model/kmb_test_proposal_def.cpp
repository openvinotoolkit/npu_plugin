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

#include "kmb_test_proposal_def.hpp"
#include "kmb_test_add_def.hpp"

#include <ngraph/op/experimental/layers/proposal.hpp>

#include <blob_factory.hpp>

namespace {
    void generate_anchors(int base_size, const std::vector<float>& ratios,
                          const std::vector<float>& scales, float* anchors, float coordinates_offset,
                          bool shift_anchors, bool round_ratios) {
        // base box's width & height & center location
        const float base_area = static_cast<float>(base_size * base_size);
        const float half_base_size = base_size * 0.5f;
        const float center = 0.5f * (base_size - coordinates_offset);

        const int num_ratios = ratios.size();
        const int num_scales = scales.size();

        // enumerate all transformed boxes
        for (int ratio = 0; ratio < num_ratios; ++ratio) {
            // transformed width & height for given ratio factors
            float ratio_w = round_ratios ? std::roundf(std::sqrt(base_area / ratios[ratio]))
                                         : std::sqrt(base_area / ratios[ratio]);

            float ratio_h = round_ratios ? std::roundf(ratio_w * ratios[ratio])
                                         : ratio_w * ratios[ratio];

            float * const p_anchors_wm = anchors + 0 * num_ratios * num_scales + ratio * num_scales;
            float * const p_anchors_hm = anchors + 1 * num_ratios * num_scales + ratio * num_scales;
            float * const p_anchors_wp = anchors + 2 * num_ratios * num_scales + ratio * num_scales;
            float * const p_anchors_hp = anchors + 3 * num_ratios * num_scales + ratio * num_scales;

            for (int scale = 0; scale < num_scales; ++scale) {
                // transformed width & height for given scale factors
                const float scale_w = 0.5f * (ratio_w * scales[scale] - coordinates_offset);
                const float scale_h = 0.5f * (ratio_h * scales[scale] - coordinates_offset);

                // (x1, y1, x2, y2) for transformed box
                p_anchors_wm[scale] = center - scale_w;
                p_anchors_hm[scale] = center - scale_h;
                p_anchors_wp[scale] = center + scale_w;
                p_anchors_hp[scale] = center + scale_h;

                if (shift_anchors) {
                    p_anchors_wm[scale] -= half_base_size;
                    p_anchors_hm[scale] -= half_base_size;
                    p_anchors_wp[scale] -= half_base_size;
                    p_anchors_hp[scale] -= half_base_size;
                }
            }
        }
    }

    void enumerate_proposals_cpu(const float* bottom4d, const float* d_anchor4d, const float* anchors,
                                 float* proposals, const int num_anchors, const int bottom_H,
                                 const int bottom_W, const float img_H, const float img_W,
                                 const float min_box_H, const float min_box_W, const int feat_stride,
                                 const float box_coordinate_scale, const float box_size_scale,
                                 float coordinates_offset, bool initial_clip, bool swap_xy, bool clip_before_nms) {
        const int    bottom_area  = bottom_H * bottom_W;
        const float* p_anchors_wm = anchors + 0 * num_anchors;
        const float* p_anchors_hm = anchors + 1 * num_anchors;
        const float* p_anchors_wp = anchors + 2 * num_anchors;
        const float* p_anchors_hp = anchors + 3 * num_anchors;

        for (int h = 0; h < bottom_H; h++)
        {
            for (int w = 0; w < bottom_W; w++)
            {
                const float x = static_cast<float>((swap_xy ? h : w) * feat_stride);
                const float y = static_cast<float>((swap_xy ? w : h) * feat_stride);

                const float* p_box   = d_anchor4d + h * bottom_W + w;
                const float* p_score = bottom4d   + h * bottom_W + w;

                float* p_proposal = proposals + (h * bottom_W + w) * num_anchors * 5;

                for (int anchor = 0; anchor < num_anchors; ++anchor)
                {
                    const float dx = p_box[(anchor * 4 + 0) * bottom_area] / box_coordinate_scale;
                    const float dy = p_box[(anchor * 4 + 1) * bottom_area] / box_coordinate_scale;

                    const float d_log_w = p_box[(anchor * 4 + 2) * bottom_area] / box_size_scale;
                    const float d_log_h = p_box[(anchor * 4 + 3) * bottom_area] / box_size_scale;

                    const float score = p_score[anchor * bottom_area];

                    float x0 = x + p_anchors_wm[anchor];
                    float y0 = y + p_anchors_hm[anchor];
                    float x1 = x + p_anchors_wp[anchor];
                    float y1 = y + p_anchors_hp[anchor];

                    if (initial_clip)
                    {
                        // adjust new corner locations to be within the image region
                        x0 = std::max<float>(0.0f, std::min<float>(x0, img_W));
                        y0 = std::max<float>(0.0f, std::min<float>(y0, img_H));
                        x1 = std::max<float>(0.0f, std::min<float>(x1, img_W));
                        y1 = std::max<float>(0.0f, std::min<float>(y1, img_H));
                    }

                    // width & height of box
                    const float ww = x1 - x0 + coordinates_offset;
                    const float hh = y1 - y0 + coordinates_offset;
                    // center location of box
                    const float ctr_x = x0 + 0.5f * ww;
                    const float ctr_y = y0 + 0.5f * hh;

                    // new center location according to gradient (dx, dy)
                    const float pred_ctr_x = dx * ww + ctr_x;
                    const float pred_ctr_y = dy * hh + ctr_y;
                    // new width & height according to gradient d(log w), d(log h)
                    const float pred_w = std::exp(d_log_w) * ww;
                    const float pred_h = std::exp(d_log_h) * hh;

                    // update upper-left corner location
                    x0 = pred_ctr_x - 0.5f * pred_w;
                    y0 = pred_ctr_y - 0.5f * pred_h;
                    // update lower-right corner location
                    x1 = pred_ctr_x + 0.5f * pred_w;
                    y1 = pred_ctr_y + 0.5f * pred_h;

                    // adjust new corner locations to be within the image region,
                    if (clip_before_nms)
                    {
                        x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
                        y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
                        x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
                        y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));
                    }

                    // recompute new width & height
                    const float box_w = x1 - x0 + coordinates_offset;
                    const float box_h = y1 - y0 + coordinates_offset;

                    p_proposal[5*anchor + 0] = x0;
                    p_proposal[5*anchor + 1] = y0;
                    p_proposal[5*anchor + 2] = x1;
                    p_proposal[5*anchor + 3] = y1;
                    p_proposal[5*anchor + 4] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;
                }
            }
        }

    }

    void unpack_boxes(const float* p_proposals, float* unpacked_boxes, int pre_nms_topn)
    {
        for (int i = 0; i < pre_nms_topn; i++)
        {
            unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[5 * i + 0];
            unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[5 * i + 1];
            unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[5 * i + 2];
            unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[5 * i + 3];
        }
    }

    void nms_cpu(const int num_boxes, int is_dead[],
                 const float* boxes, int index_out[], int* const num_out,
                 const int base_index, const float nms_thresh, const int max_num_out,
                 float coordinates_offset)
    {
        const int num_proposals = num_boxes;
        int count = 0;

        const float* x0 = boxes + 0 * num_proposals;
        const float* y0 = boxes + 1 * num_proposals;
        const float* x1 = boxes + 2 * num_proposals;
        const float* y1 = boxes + 3 * num_proposals;

        memset(is_dead, 0, num_boxes * sizeof(int));

        for (int box = 0; box < num_boxes; ++box)
        {
            if (is_dead[box])
                continue;

            index_out[count++] = base_index + box;
            if (count == max_num_out)
                break;

            int tail = box + 1;

            for (; tail < num_boxes; ++tail)
            {
                float res = 0.0f;

                const float x0i = x0[box];
                const float y0i = y0[box];
                const float x1i = x1[box];
                const float y1i = y1[box];

                const float x0j = x0[tail];
                const float y0j = y0[tail];
                const float x1j = x1[tail];
                const float y1j = y1[tail];

                if (x0i <= x1j && y0i <= y1j && x0j <= x1i && y0j <= y1i)
                {
                    // overlapped region (= box)
                    const float x0 = std::max<float>(x0i, x0j);
                    const float y0 = std::max<float>(y0i, y0j);
                    const float x1 = std::min<float>(x1i, x1j);
                    const float y1 = std::min<float>(y1i, y1j);

                    // intersection area
                    const float width  = std::max<float>(0.0f,  x1 - x0 + coordinates_offset);
                    const float height = std::max<float>(0.0f,  y1 - y0 + coordinates_offset);
                    const float area   = width * height;

                    // area of A, B
                    const float A_area = (x1i - x0i + coordinates_offset) * (y1i - y0i + coordinates_offset);
                    const float B_area = (x1j - x0j + coordinates_offset) * (y1j - y0j + coordinates_offset);

                    // IoU
                    res = area / (A_area + B_area - area);
                }

                if (nms_thresh < res)
                    is_dead[tail] = 1;
            }
        }

        *num_out = count;
    }

    void retrieve_rois_cpu(const int num_rois, const int item_index,
                           const int num_proposals,
                           const float* proposals, const int roi_indices[],
                           float* rois, int post_nms_topn_,
                           bool normalize, float img_h, float img_w, bool clip_after_nms)
    {
        const float *src_x0    = proposals + 0 * num_proposals;
        const float *src_y0    = proposals + 1 * num_proposals;
        const float *src_x1    = proposals + 2 * num_proposals;
        const float *src_y1    = proposals + 3 * num_proposals;

        for (int roi = 0; roi < num_rois; roi++)
        {
            int index = roi_indices[roi];

            float x0 = src_x0[index];
            float y0 = src_y0[index];
            float x1 = src_x1[index];
            float y1 = src_y1[index];

            if (clip_after_nms)
            {
                x0 = std::max<float>(0.0f, std::min<float>(x0, img_w));
                y0 = std::max<float>(0.0f, std::min<float>(y0, img_h));
                x1 = std::max<float>(0.0f, std::min<float>(x1, img_w));
                y1 = std::max<float>(0.0f, std::min<float>(y1, img_h));
            }

            if (normalize)
            {
                x0 /= img_w;
                y0 /= img_h;
                x1 /= img_w;
                y1 /= img_h;
            }

            rois[roi * 5 + 0] = static_cast<float>(item_index);
            rois[roi * 5 + 1] = x0;
            rois[roi * 5 + 2] = y0;
            rois[roi * 5 + 3] = x1;
            rois[roi * 5 + 4] = y1;
        }

        if (num_rois < post_nms_topn_)
        {
            for (int i = 5 * num_rois; i < 5 * post_nms_topn_; i++)
            {
                rois[i] = 0.f;
            }

            // marker at end of boxes list
            rois[num_rois * 5 + 0] = -1;
        }
    }

    BlobVector refProposal(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
        if (inputs.size() != 3) {
            THROW_IE_EXCEPTION << "Incorrect number of inputs";
        }

        const auto proposal = std::dynamic_pointer_cast<ngraph::op::Proposal>(layer);
        const auto& attrs = proposal->get_attrs();

        int anchors_shape_0 = attrs.ratio.size() * attrs.scale.size();
        std::vector<float> anchors(anchors_shape_0 * 4);

        IE_ASSERT(attrs.framework == "caffe" || attrs.framework == "tensorflow");
        // For framework == "tensorflow"
        float coordinates_offset = 0.0f;
        bool initial_clip        = true;
        bool shift_anchors       = true;
        bool round_ratios        = false;
        bool swap_xy             = true;

        if (attrs.framework == "caffe") {
            coordinates_offset = 1.0f;
            initial_clip = false;
            shift_anchors = false;
            round_ratios = true;
            swap_xy = false;
        }

        generate_anchors(attrs.base_size, attrs.ratio, attrs.scale, &anchors[0],
                         coordinates_offset, shift_anchors, round_ratios);

        std::vector<int> roi_indices(attrs.post_nms_topn);

        // Allocate output
        auto shape = layer->get_shape();
        IE_ASSERT(shape.size() == 2);
        SizeVector outDims = {shape[0], shape[1]};

        auto outDesc = TensorDesc(Precision::FP32, outDims, Layout::NC);
        auto output  = make_blob_with_precision(outDesc);
        output->allocate();

        // Prepare memory
        const float *p_bottom_item   = inputs[0]->buffer();
        const float *p_d_anchor_item = inputs[1]->buffer();
        const float *p_img_info_cpu  = inputs[2]->buffer();
        float *p_roi_item            = output->buffer();

        /* FIXME: Implement reshape layer in test framework
         * Usually image info has the shape {1, C} then the network uses a reshape,
         * because ngraph for proposal layer uses the shape {C}, but since the reshape layer
         * in test framework isn't implemented yet, we create image info immediately with the shape {C}
         * Uncomment after implementing reshape layer in test framework:
         * size_t img_info_size = inputs[2]->getTensorDesc().getDims()[1];
         */
        size_t img_info_size = inputs[2]->getTensorDesc().getDims()[0];

        // bottom shape: (2 x num_anchors) x H x W
        const int bottom_H = inputs[0]->getTensorDesc().getDims()[2];
        const int bottom_W = inputs[0]->getTensorDesc().getDims()[3];

        // input image height & width
        const float img_H = p_img_info_cpu[swap_xy ? 1 : 0];
        const float img_W = p_img_info_cpu[swap_xy ? 0 : 1];

        // scale factor for height & width
        const float scale_H = p_img_info_cpu[2];
        const float scale_W = img_info_size > 3 ? p_img_info_cpu[3] : scale_H;

        // minimum box width & height
        const float min_box_H = attrs.min_size * scale_H;
        const float min_box_W = attrs.min_size * scale_W;

        // number of all proposals = num_anchors * H * W
        const int num_proposals = anchors_shape_0 * bottom_H * bottom_W;

        // number of top-n proposals before NMS
        const int pre_nms_topn = std::min<int>(num_proposals, attrs.pre_nms_topn);

        // number of final RoIs
        int num_rois = 0;

        // enumerate all proposals
        //   num_proposals = num_anchors * H * W
        //   (x1, y1, x2, y2, score) for each proposal
        // NOTE: for bottom, only foreground scores are passed
        struct ProposalBox {
            float x0;
            float y0;
            float x1;
            float y1;
            float score;
        };

        std::vector<ProposalBox> proposals(num_proposals);
        const int unpacked_boxes_buffer_size = 4 * pre_nms_topn;
        std::vector<float> unpacked_boxes(unpacked_boxes_buffer_size);
        std::vector<int> is_dead(pre_nms_topn);

        // Execute
        int nn = inputs[0]->getTensorDesc().getDims()[0];
        for (int n = 0; n < nn; ++n) {
            enumerate_proposals_cpu(p_bottom_item + num_proposals + n * num_proposals * 2,
                                    p_d_anchor_item + n * num_proposals * 4,
                                    &anchors[0], reinterpret_cast<float*>(&proposals[0]),
                                    anchors_shape_0, bottom_H, bottom_W, img_H, img_W,
                                    min_box_H, min_box_W, attrs.feat_stride,
                                    attrs.box_coordinate_scale, attrs.box_size_scale,
                                    coordinates_offset, initial_clip, swap_xy, attrs.clip_before_nms);
            std::partial_sort(proposals.begin(), proposals.begin() + pre_nms_topn, proposals.end(),
                    [](const ProposalBox &struct1, const ProposalBox &struct2) {
                        return (struct1.score > struct2.score);
                    });

            unpack_boxes(reinterpret_cast<float*>(&proposals[0]), &unpacked_boxes[0], pre_nms_topn);
            nms_cpu(pre_nms_topn, &is_dead[0], &unpacked_boxes[0], &roi_indices[0], &num_rois, 0,
                    attrs.nms_thresh, attrs.post_nms_topn, coordinates_offset);

            retrieve_rois_cpu(num_rois, n, pre_nms_topn, &unpacked_boxes[0], &roi_indices[0],
                              p_roi_item + n * attrs.post_nms_topn * 5,
                              attrs.post_nms_topn, attrs.normalize, img_H, img_W, attrs.clip_after_nms);
        }
        return {output};
    }
}  // namespace

TestNetwork& ProposalLayerDef::build() {
    ngraph::op::ProposalAttrs attr;
    attr.base_size            = params_.base_size_;
    attr.nms_thresh           = params_.nms_thresh_;
    attr.feat_stride          = params_.feat_stride_;
    attr.min_size             = params_.min_size_;
    attr.pre_nms_topn         = params_.pre_nms_topn_;
    attr.post_nms_topn        = params_.post_nms_topn_;
    attr.ratio                = params_.ratio_;
    attr.scale                = params_.scale_;
    attr.framework            = params_.framework_;
    attr.clip_after_nms       = params_.clip_after_nms_;
    attr.normalize            = params_.normalize_;
    attr.box_size_scale       = params_.box_size_scale_;
    attr.box_coordinate_scale = params_.box_coordinate_scale_;

    auto cls_score = net_.getPort(cls_score_port_);
    auto bbox_pred = net_.getPort(bbox_pred_port_);
    auto img_info  = net_.getPort(img_info_port_);
    auto proposal  = std::make_shared<ngraph::op::Proposal>(cls_score, bbox_pred, img_info, attr);

    return net_.addLayer(name_, proposal, refProposal);
}
