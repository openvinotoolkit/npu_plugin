# Image Detection C++ Sample
This sample will detect object by YOLO V2 tiny model. Input model blob and nv12 image, inference and then get bounding box of object and box's confidence.

## Running
```sh
./kmb_detection_sample -m [blob file] -i [nv12 file] -iw [width of nv12 image] -ih [height of nv12 image]
```
Blobs are strored in the https://gitlab-icv.inn.intel.com/inference-engine/models-ir repository.

Any blob from `$(MODELS-IR-ROOT)/KMB_models/BLOBS/` can be used for this sample.
