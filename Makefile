PULP_APP = mrbci

PULP_APP_FC_SRCS = \
    src/fc/main.c

PULP_APP_CL_SRCS = \
    src/cl/cluster.c \
	src/cl/input.c \
	src/cl/func/convert.c \
	src/cl/func/copy_mat.c \
	src/cl/func/covmat.c \
	src/cl/func/matmul.c \
	src/cl/func/sos_filt.c \
	src/cl/func/swap_mat.c \
	src/cl/linalg/matop_f.c \
	src/cl/linalg/svd.c \
	src/cl/linalg/svd_parallel.c \
	src/cl/mrbci/covmat.c \
	src/cl/mrbci/feature_extraction.c \
	src/cl/mrbci/filter.c \
	src/cl/mrbci/half_diag.c \
	src/cl/mrbci/logm.c \
	src/cl/mrbci/mrbci.c \
	src/cl/mrbci/mrbci_params.c \
	src/cl/mrbci/svm.c \
	src/cl/mrbci/whitening.c \

PULP_CFLAGS = -O3 -g

PULP_LDFLAGS += -lplpdsp -lm

# enable slow householder
# PULP_CFLAGS += -DHOUSEHOLDER_SLOW

# enable parallel computation
PULP_CFLAGS += -DPARALLEL

# enable FMA support
PULP_CFLAGS += -DUSE_FUSED_FPU

# disable square root unit
# PULP_CFLAGS += -DUSE_SOFT_SQRTDIV
# PULP_CFLAGS += -mno-fdiv

# do Power Measurement
# PULP_CFLAGS += "-DPOWER"

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
