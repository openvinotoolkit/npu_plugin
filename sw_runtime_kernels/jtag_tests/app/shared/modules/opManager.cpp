// {% copyright %}

#include "opManager.h"
//#include "ConvND.h"
//#include "Convolution.h"
//#include "Pooling.h"
//#include "PoolND.h"
//#include "FCL.h"
//#include "Softmax.h"
//#include "Deconvolution.h"
#include "NoOp.h"
//#include "lrn.h"
//#include "EltWise.h"
//#include "Convert.h"
#include "Permute.h"
//#include "DetectionOutput.h"
//#include "Tile.h"
//#include "Normalize.h"
//#include "RegionYolo.h"
//#include "ReorgYolo.h"
//#include "CTCDecoder.h"
//#include "HwFcRelayout.h"
//#include "Im2ColConvolution.h"
//#include "Grn.h"
//#include "Mvn.h"
//#include "Proposal.h"
//#include "Psroipooling.h"
//#include "ROIPooling.h"
//#include "Interp.h"
//#include "Dummy.h"
//#include "Custom.h"
//#include "CustomOcl.h"
#include "CustomCpp.h"
//#include "CustomDMA.h"
//#include "mvFaceDetector.h"
//#include "LSTMCell.h"
//#include "Copy.h"
//#include "Pad.h"
//#include "Resample.h"
//#include "Upsampling.h"
//#include "ArgMax.h"
//#include "PostOps.h"
//#include "GEMM.h"
//#include "Reduce.h"
//#include "ReverseSequence.h"
//#include "Gather.h"
//#include "SCRelu.h"
//#include "Scatter.h"
//#include "TopK.h"
//#include "ExpDetectionOutput.h"
//#include "StaticShapeNMS.h"
//#include "ROIFeatureExtractor.h"
//#include "OneHot.h"
//#include "LoopStart.h"
//#include "LoopEnd.h"
//#include "ExpPriorGridGenerator.h"
#include "PostOps.h"
//#include "FakeQuantize.h"
//#include "Pad.h"
//#include "Interpolate.h"
//#include "CTCGreedyDecoderSeqLen.h"
//#include "SpaceToDepth.h"
//#include "GatherElements.h"
//#include "DepthToSpace.h"
//#include "StridedSlice.h"
//#include "GatherND.h"
//#include "SWConvolution.h"
//#include "ScatterElementsUpdate.h"
//#include "ScatterUpdate.h"
//#include "OutShapeOfReshape.h"
//#include "Upsampling.h"
//#include "NonZero.h"
//#include "Broadcast.h"
//#include "InnerLRN.h"
//#include "ExpDetectionOutput.h"
//#include "ExpGenerateProposals.h"
//#include "ExpPriorGridGenerator.h"
//#include "ExpTopKROIs.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MA2480
#include "HwOp.h"
#endif

Op * opManager::createOp(t_MvTensorOpType which_one, int OpPosition, int /*numberOfNCEs*/) {

    if (OpPosition == primaryOperation) {
        switch(which_one) {
/*        case kPool:
            return new Pooling(which_one);
        case kSoftMax:
            return new Softmax(which_one);
        case kROIPooling:
            return new ROIPooling(which_one);
        case kPSROIPooling:
            return new PSROIPooling(which_one);
        case kDetectionOutput:
            return new DetectionOutput(which_one);
        case kDeconvolution:
            return new Deconvolution(which_one);
        case kConv:
            return new Convolution(which_one);
        case kFC:
            return new FC(which_one);
        case kCTCDecoder:
            return new CTCDecoder(which_one);
        case kGRN:
            return new GRN(which_one);
        case kMVN:
            return new MVN(which_one);
        case kNormalize:
            return new Normalize(which_one);
        case kLRN:
            return new LRN(which_one);
        case kRegionYolo:
            return new RegionYolo(which_one);
        case kProposal:
            return new Proposal(which_one);
        case kInterp:
            return new Interp(which_one);
        case kTile:
            return new Tile(which_one);
        case kGather:
             return new Gather(which_one);
        case kResample:
            return new Resample(which_one);*/
        case kClamp:
        case kElu:
        case kPower:
        case kBiasLeakyRelu:
        case kLeakyRelu:
        case kBiasRelu:
        case kRelu:
        case kPRelu:
        case kSigmoidPostop:
        case kTanh:
        case kBias:
        case kScale:
        case kScaleShift:
        case kHSwish:
        case kSwish:
        case kSoftPlus:
        case kMish:
        case kFloor:
        case kRound:
        case kErf:
        case kCeiling:
        case kGelu:
        case kLog:
        case kExp:
            return new PostOps(which_one);
/*        case kSum:
        case kProd:
        case kMax:
        case kDiv:
        case kMin:
        case kSqdiff:
        case kCompareEQ:
        case kCompareNE:
        case kCompareGT:
        case kCompareGE:
        case kCompareLT:
        case kCompareLE:
        case kLogicalNOT:
        case kLogicalAND:
        case kLogicalOR:
        case kLogicalXOR:
        case kPow:
        case kFloorMod:
        case kSelect:
            return new EltWise(which_one);*/
        case kPermute:
            return new Permute(which_one);
/*        case kConvert:
            return new Convert(which_one);
        case kFakeQuantize:
            return new FakeQuantize(which_one);
        case kPad:
            return new Pad(which_one);
        case kInterpolate:
            return new Interpolate(which_one);
        case kCTCGreedyDecoderSeqLen:
            return new CTCGreedyDecoderSeqLen(which_one);
        case kSpaceToDepth:
            return new SpaceToDepth(which_one);
        case kDummy:
            return new Dummy(which_one);
        case kGatherElements:
             return new GatherElements(which_one);
        case kDepthToSpace:
            return new DepthToSpace(which_one);
        case kReverseSequence:
            return new ReverseSequence(which_one);
        case kCustomOcl:
            return new CustomOcl(which_one);*/
        case kCustomCpp:
            return new CustomCpp(which_one);
/*        case kStridedSlice:
            return new StridedSlice(which_one);
        case kLSTMCell:
            return new LSTMCell(which_one);
        case kGatherND:
            return new GatherND(which_one);
        case kSWConvolution:
            return new SWConvolution(which_one);
        case kScatterElementsUpdate:
            return new ScatterElementsUpdate(which_one);
        case kScatterUpdate:
            return new ScatterUpdate(which_one);
        case kOutShapeOfReshape:
            return new OutShapeOfReshape(which_one);
        case kUpsampling:
            return new Upsampling(which_one);
        case kNonZero:
            return new NonZero(which_one);
        case kBroadcast:
            return new Broadcast(which_one);
        case kInnerLRN:
            return new InnerLRN(which_one);
        case kNMS:
            return new StaticShapeNMS(which_one);
        case kExpDetectionOutput:
            return new ExpDetectionOutput(which_one);
        case kExpGenerateProposals:
            return new ExpGenerateProposals(which_one);
        case kExpPriorGridGenerator:
            return new ExpPriorGridGenerator(which_one);
        case kExpTopKROIs:
            return new ExpTopKROIs(which_one);*/
        default:
            printf("NO SUCH Op STAGE: %i\n", which_one);
            return new NoOp(kNone0);
        }
    } else {
        printf("No SUCH STAGE\n");
        return new NoOp(kNone0);
    }
}
