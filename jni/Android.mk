LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

include /home/shushman/Documents/opencv4android/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := test_op3
LOCAL_SRC_FILES := image.cpp 
LOCAL_LDLIBS +=  -llog -ldl -landroid

include $(BUILD_SHARED_LIBRARY)
