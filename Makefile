PULP_APP = eegnet

PULP_APP_FC_SRCS = \
    src/fc/main.c

PULP_APP_CL_SRCS = \
    src/cl/cluster.c \
	src/cl/func/sos_filt.c \

PULP_CFLAGS = -O3 -g -DROUND

PULP_LDFLAGS += -lplpdsp

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
