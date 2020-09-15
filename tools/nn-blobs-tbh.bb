SUMMARY = "NN BLOBS (TBH) FOR NN/VPUAL TEST APPS"
DESCRIPTION = "Adding the NN Blobs required by NN/VPUAL test apps"
LICENSE = "CLOSED"

RDEPENDS_${PN} = "kernel-module-udmabuf openvino"

SRC_URI = "file://${NN_BLOBS_PATH}/nn-blobs-tbh.tar.bz2"

PV = "nn_blobs_75e0804+NN_Compiler_v1.4.9"

S = "${WORKDIR}/release_blobs"

SOLIBS = ".so"
FILES_SOLIBSDEV = ""

INSANE_SKIP_${PN} = "ldflags"

do_install () {
    install -d ${D}/opt/facenet-20180408-102900_ov
    install -m 0755 ${S}/facenet-20180408-102900_ov/expected_result_sim.dat ${D}/opt/facenet-20180408-102900_ov
    install -m 0755 ${S}/facenet-20180408-102900_ov/input.dat ${D}/opt/facenet-20180408-102900_ov
    install -m 0755 ${S}/facenet-20180408-102900_ov/facenet-20180408-102900_ov.blob ${D}/opt/facenet-20180408-102900_ov
    install -d ${D}/opt/googlenet-v1_ov
    install -m 0755 ${S}/googlenet-v1_ov/expected_result_sim.dat ${D}/opt/googlenet-v1_ov
    install -m 0755 ${S}/googlenet-v1_ov/input.dat ${D}/opt/googlenet-v1_ov
    install -m 0755 ${S}/googlenet-v1_ov/googlenet-v1_ov.blob ${D}/opt/googlenet-v1_ov
    install -d ${D}/opt/googlenet-v3_ov
    install -m 0755 ${S}/googlenet-v3_ov/expected_result_sim.dat ${D}/opt/googlenet-v3_ov
    install -m 0755 ${S}/googlenet-v3_ov/input.dat ${D}/opt/googlenet-v3_ov
    install -m 0755 ${S}/googlenet-v3_ov/googlenet-v3_ov.blob ${D}/opt/googlenet-v3_ov
    install -d ${D}/opt/googlenet-v4_ov
    install -m 0755 ${S}/googlenet-v4_ov/expected_result_sim.dat ${D}/opt/googlenet-v4_ov
    install -m 0755 ${S}/googlenet-v4_ov/input.dat ${D}/opt/googlenet-v4_ov
    install -m 0755 ${S}/googlenet-v4_ov/googlenet-v4_ov.blob ${D}/opt/googlenet-v4_ov
    install -d ${D}/opt/mobilenet_ov
    install -m 0755 ${S}/mobilenet_ov/expected_result_sim.dat ${D}/opt/mobilenet_ov
    install -m 0755 ${S}/mobilenet_ov/input.dat ${D}/opt/mobilenet_ov
    install -m 0755 ${S}/mobilenet_ov/mobilenet_ov.blob ${D}/opt/mobilenet_ov
    install -d ${D}/opt/mobilenet-v1-1.0-224_ov
    install -m 0755 ${S}/mobilenet-v1-1.0-224_ov/expected_result_sim.dat ${D}/opt/mobilenet-v1-1.0-224_ov
    install -m 0755 ${S}/mobilenet-v1-1.0-224_ov/input.dat ${D}/opt/mobilenet-v1-1.0-224_ov
    install -m 0755 ${S}/mobilenet-v1-1.0-224_ov/mobilenet-v1-1.0-224_ov.blob ${D}/opt/mobilenet-v1-1.0-224_ov
    install -d ${D}/opt/resnet-18-pytorch_ov
    install -m 0755 ${S}/resnet-18-pytorch_ov/expected_result_sim.dat ${D}/opt/resnet-18-pytorch_ov
    install -m 0755 ${S}/resnet-18-pytorch_ov/input.dat ${D}/opt/resnet-18-pytorch_ov
    install -m 0755 ${S}/resnet-18-pytorch_ov/resnet-18-pytorch_ov.blob ${D}/opt/resnet-18-pytorch_ov
    install -d ${D}/opt/resnet-101_ov
    install -m 0755 ${S}/resnet-101_ov/expected_result_sim.dat ${D}/opt/resnet-101_ov
    install -m 0755 ${S}/resnet-101_ov/input.dat ${D}/opt/resnet-101_ov
    install -m 0755 ${S}/resnet-101_ov/resnet-101_ov.blob ${D}/opt/resnet-101_ov
    install -d ${D}/opt/resnet-152_ov
    install -m 0755 ${S}/resnet-152_ov/expected_result_sim.dat ${D}/opt/resnet-152_ov
    install -m 0755 ${S}/resnet-152_ov/input.dat ${D}/opt/resnet-152_ov
    install -m 0755 ${S}/resnet-152_ov/resnet-101_ov.blob ${D}/opt/resnet-152_ov
    install -d ${D}/opt/resnet_ov
    install -m 0755 ${S}/resnet_ov/expected_result_sim.dat ${D}/opt/resnet_ov
    install -m 0755 ${S}/resnet_ov/input.dat ${D}/opt/resnet_ov
    install -m 0755 ${S}/resnet_ov/resnet_ov.blob ${D}/opt/resnet_ov
    install -d ${D}/opt/squeezenet1.1_ov
    install -m 0755 ${S}/squeezenet1.1_ov/expected_result_sim.dat ${D}/opt/squeezenet1.1_ov
    install -m 0755 ${S}/squeezenet1.1_ov/input.dat ${D}/opt/squeezenet1.1_ov
    install -m 0755 ${S}/squeezenet1.1_ov/squeezenet1.1_ov.blob ${D}/opt/squeezenet1.1_ov
    install -d ${D}/opt/ssd-mobilenet-v1-coco_ov
    install -m 0755 ${S}/ssd-mobilenet-v1-coco_ov/expected_result_sim.dat ${D}/opt/ssd-mobilenet-v1-coco_ov
    install -m 0755 ${S}/ssd-mobilenet-v1-coco_ov/input.dat ${D}/opt/ssd-mobilenet-v1-coco_ov
    install -m 0755 ${S}/ssd-mobilenet-v1-coco_ov/ssd-mobilenet-v1-coco_ov.blob ${D}/opt/ssd-mobilenet-v1-coco_ov
    install -d ${D}/opt/tiny-yolo-v1_ov
    install -m 0755 ${S}/tiny-yolo-v1_ov/expected_result_sim.dat ${D}/opt/tiny-yolo-v1_ov
    install -m 0755 ${S}/tiny-yolo-v1_ov/input.dat ${D}/opt/tiny-yolo-v1_ov
    install -m 0755 ${S}/tiny-yolo-v1_ov/tiny-yolo-v1_ov.blob ${D}/opt/tiny-yolo-v1_ov
    install -d ${D}/opt/yolotiny_ov
    install -m 0755 ${S}/yolotiny_ov/expected_result_sim.dat ${D}/opt/yolotiny_ov
    install -m 0755 ${S}/yolotiny_ov/input.dat ${D}/opt/yolotiny_ov
    install -m 0755 ${S}/yolotiny_ov/yolotiny_ov.blob ${D}/opt/yolotiny_ov
    install -d ${D}/opt/yolo-v2-ava-0001_ov
    install -m 0755 ${S}/yolo-v2-ava-0001_ov/expected_result_sim.dat ${D}/opt/yolo-v2-ava-0001_ov
    install -m 0755 ${S}/yolo-v2-ava-0001_ov/input.dat ${D}/opt/yolo-v2-ava-0001_ov
    install -m 0755 ${S}/yolo-v2-ava-0001_ov/yolo-v2-ava-0001_ov.blob ${D}/opt/yolo-v2-ava-0001_ov
}

FILES_${PN} += " /opt"
