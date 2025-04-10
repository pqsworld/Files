# Copyright 2016 Silead(Shanghai) Inc.
#
# Android.mk for libfp
#

# =========================================================

LOCAL_PATH:= $(call my-dir)
API_PATH = ..
NET_C_INCLUDES := $(LOCAL_PATH)/$(API_PATH)/net/

NET_SRC_FILES := \
  $(API_PATH)/net/net_cnn_common.c\
  $(API_PATH)/net/SL_Math.c\
  $(API_PATH)/net/android.c\
  $(API_PATH)/net/block_function.c\

ifeq ($(PROJECT),PROJECT_SQUARE)
 NET_C_INCLUDES += $(LOCAL_PATH)/$(API_PATH)/net/6135/
#echo "square not support net"
 NET_SRC_FILES += \
    $(API_PATH)/net/6135/alog.c\
	$(API_PATH)/net/6135/net_spoof.c\
	$(API_PATH)/net/6135/net_mistouch.c
endif
ifeq ($(PROJECT),PROJECT_6157)
 NET_C_INCLUDES += $(LOCAL_PATH)/$(API_PATH)/net/6157/
 NET_SRC_FILES += \
    $(API_PATH)/net/6157/alog.c\
	$(API_PATH)/net/6157/net_mistouch.c\
	$(API_PATH)/net/6157/net_spoof.c
endif
ifeq ($(PROJECT),PROJECT_6159)
 NET_C_INCLUDES += $(LOCAL_PATH)/$(API_PATH)/net/6159/
 NET_SRC_FILES += \
    $(API_PATH)/net/6159/alog.c\
	$(API_PATH)/net/6159/net_mistouch.c\
	$(API_PATH)/net/6159/net_spoof.c
endif
ifeq ($(PROJECT),PROJECT_6191)
 NET_C_INCLUDES += $(LOCAL_PATH)/$(API_PATH)/net/6191/
 NET_SRC_FILES += \
    $(API_PATH)/net/6191/alog.c\
	$(API_PATH)/net/6191/net_mistouch.c\
	$(API_PATH)/net/6191/net_spoof.c
endif
ifeq ($(PROJECT),PROJECT_6192)
 NET_C_INCLUDES += $(LOCAL_PATH)/$(API_PATH)/net/6192/
 NET_SRC_FILES += \
    $(API_PATH)/net/6192/alog.c\
	$(API_PATH)/net/6192/net_ori_enh.c\
	$(API_PATH)/net/6192/net_patch.c\
	$(API_PATH)/net/6192/net_spoof.c\
	$(API_PATH)/net/6192/net_mistouch.c\
 	$(API_PATH)/net/6192/net_spd.c\
 	$(API_PATH)/net/6192/net_quality.c
endif
ifeq ($(PROJECT),PROJECT_6193)
 NET_C_INCLUDES += $(LOCAL_PATH)/$(API_PATH)/net/6193/
 NET_SRC_FILES += \
    $(API_PATH)/net/6193/alog.c\
 	$(API_PATH)/net/6193/net_mistouch.c\
 	$(API_PATH)/net/6193/net_spoof.c\
 	$(API_PATH)/net/6193/net_spd.c\
 	$(API_PATH)/net/6193/net_mask.c\
 	$(API_PATH)/net/6193/net_enhance.c\
 	$(API_PATH)/net/6193/net_patch.c\
 	$(API_PATH)/net/6193/net_exp.c\
 	$(API_PATH)/net/6193/net_quality.c
endif



include $(CLEAR_VARS)
LOCAL_CFLAGS += -DMAKE_SO -Wno-error=date-time
LOCAL_CFLAGS += -D$(PROJECT) -D$(CUST)
LOCAL_MODULE := libsl_fp_algo_net
LOCAL_MODULE_TAGS := optional
LOCAL_LDFLAGS += $(foreach f, $(strip malloc free realloc calloc), -Wl,--wrap=$(f))
LOCAL_MULTILIB := both
#LOCAL_SHARED_LIBRARIES += libcutils\
                          $(NET_NAME)
LOCAL_SRC_FILES := $(NET_SRC_FILES)
LOCAL_C_INCLUDES := $(NET_C_INCLUDES)
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_CFLAGS += -DMAKE_SO -Wno-error=date-time
LOCAL_CFLAGS += -D$(PROJECT) -D$(CUST)
LOCAL_MODULE := libsl_fp_algo_net
LOCAL_MODULE_TAGS := optional
LOCAL_LDFLAGS += $(foreach f, $(strip malloc free realloc calloc), -Wl,--wrap=$(f))
LOCAL_MULTILIB := both
LOCAL_SHARED_LIBRARIES += libcutils\
													liblog
LOCAL_SRC_FILES := $(NET_SRC_FILES)
LOCAL_C_INCLUDES := $(NET_C_INCLUDES)
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_CFLAGS += -DNEON_OFF -Wno-error=date-time
LOCAL_CFLAGS += -D$(PROJECT) -D$(CUST)
LOCAL_MODULE := libsl_fp_algo_net
LOCAL_MODULE_HOST_OS := linux #windows
LOCAL_IS_HOST_MODULE := true
LOCAL_MODULE_TAGS := optional
LOCAL_LDFLAGS += $(foreach f, $(strip malloc free realloc calloc), -Wl,--wrap=$(f))
LOCAL_MULTILIB := both
LOCAL_SHARED_LIBRARIES += libcutils\
													liblog
LOCAL_SRC_FILES := $(NET_SRC_FILES)
LOCAL_C_INCLUDES := $(NET_C_INCLUDES)
include $(BUILD_HOST_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_CFLAGS += -DMAKE_SO -DNEON_OFF -Wno-error=date-time#DMAKESO
LOCAL_CFLAGS += -D$(PROJECT) -D$(CUST)
LOCAL_MODULE := libsl_fp_algo_net_mem
LOCAL_MODULE_HOST_OS := linux #windows
LOCAL_IS_HOST_MODULE := true
LOCAL_MODULE_TAGS := optional
LOCAL_LDFLAGS += $(foreach f, $(strip malloc free realloc calloc), -Wl,--wrap=$(f))
LOCAL_MULTILIB := both
#LOCAL_SHARED_LIBRARIES += libcutils\
                          $(NET_NAME)
LOCAL_SRC_FILES := $(NET_SRC_FILES)
LOCAL_C_INCLUDES := $(NET_C_INCLUDES)
include $(BUILD_HOST_SHARED_LIBRARY)

#NETL


