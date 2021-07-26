PULP_APP = test

PULP_APP_CL_SRCS = \
    cluster.c \

PULP_APP_FC_SRCS = \
    test.c \

PULP_LDFLAGS += -lplpdsp
PULP_CFLAGS = -O3 -g 

PULP_LDFLAGS += -lm

# Enable Fused FMA
#PULP_CFLAGS += -DUSE_FUSED_FPU



include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
