PULP_APP = test

PULP_APP_FC_SRCS = \
    test.c cluster.c

#PULP_LDFLAGS += -lplpdsp
PULP_CFLAGS = -O3 -g 

PULP_LDFLAGS += -lm

# Enable Fused FMA
PULP_CFLAGS += -DUSE_FUSED_FPU



include $(GAP_SDK_HOME)/tools/rules/pulp_rules.mk
