/**
 * @file mrbci.h
 * @author Tibor Schneider
 * @date 2020/02/28
 * @brief This file contains the implementation for extracting the features
 */

/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rt/rt_api.h"
#include "mrbci.h"

#ifndef HOUSEHOLDER_SLOW
#define _LOGM_WORKSPACE_SIZE (MRBCI_C * (2 * MRBCI_C + 2))
#else //HOUSEHOLDER_SLOW
#define _LOGM_WORKSPACE_SIZE (MRBCI_C * (3 * MRBCI_C + 2))
#endif

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 9
#endif//NUM_WORKERS

// struct for the kernel arguments
typedef struct {
    int32_t* p_whitened;
    float* p_logm_workspace;
    int8_t* p_logm_out;
    int8_t* p_out;
    unsigned int start_freq_idx;
} _mrbci_feature_extraction_kernel_instance_t;

// kernel for parallel computation of the logm
void _mrbci_feature_extraction_kernel(void* args) {

    unsigned int _core_id = rt_core_id();

    // read input parameters
    _mrbci_feature_extraction_kernel_instance_t* _args = args;

    int32_t* _p_whitened = _args->p_whitened;
    float* _p_logm_workspace = _args->p_logm_workspace;
    int8_t* _p_logm_out = _args->p_logm_out;
    int8_t* _p_out = _args->p_out;
    unsigned int _freq_idx = _args->start_freq_idx + _core_id;

    // only compute if the frequency idx is in the allowed range
    if (_freq_idx < MRBCI_NUM_FREQ) {

        // update pointers to use the data for the specific thread
        _p_whitened += _core_id * MRBCI_C * MRBCI_C;
        _p_logm_workspace += _core_id * _LOGM_WORKSPACE_SIZE;
        _p_logm_out += _core_id * MRBCI_C * MRBCI_C_ALIGN;
        _p_out += _core_id * MRBCI_HALF_DIAG_FEATURES;

        // compute matrix logarithm
        mrbci_logm(_p_whitened, _freq_idx, _p_logm_out, _p_logm_workspace);

        // compute half diagonalization
        mrbci_half_diag(_p_logm_out, _p_out);

    }

    rt_team_barrier();
}


#endif//PARALLEL

/**
 * @brief compute all features from the input
 *
 * @warning p_in and p_out is not allowed to be in L1, it must reside in L2 memory.
 * 
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param p_out Pointer to the entire result vector (of shape MRBCI_SVM_NUM_FEATURES) in L2.
 */
void mrbci_extract_features(const int8_t* p_in,
                            int8_t* p_out) {

#ifdef PARALLEL

    rt_dma_copy_t _copy;

    /*
     * First part: Compute logm input for all different frequency bands, one at a time
     */

    // allocate the data the intermediate logm input data
    int32_t* _p_whitened_l2 = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int32_t) * MRBCI_NUM_FREQ * MRBCI_C * MRBCI_C);
    int32_t* _p_whitened_l2_iter = _p_whitened_l2;

    // allocate the input
    int8_t* _p_input_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // allocate the space for the logm input computation
    int8_t* _p_filter_data_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);
    int16_t* _p_covmat_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int16_t) * MRBCI_C * MRBCI_C);
    int32_t* _p_whitened_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C);
    int32_t* _p_whitening_workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // copy the input data
    rt_dma_memcpy((unsigned int)p_in,
                  (unsigned int)_p_input_l1,
                  sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);

    // compute the logm input for all frequencies
    for (unsigned int _freq_idx = 0; _freq_idx < MRBCI_NUM_FREQ; _freq_idx++) {

        // fill the last 4 columns with 0s to pad it correctly
        for (unsigned int _ch = 0; _ch < MRBCI_C; _ch++) {
            *((int32_t*)(_p_filter_data_l1 + MRBCI_T_ALIGN * _ch + MRBCI_T_ALIGN - 4)) = 0;
        }
        
        // wait until the input dma is finished in the first iteration
        if (_freq_idx == 0) {
            rt_dma_wait(&_copy);
        }

        // apply the filtering
        mrbci_filter(_p_input_l1, _freq_idx, _p_filter_data_l1);

        // compute the covariance matrix
        mrbci_covmat(_p_filter_data_l1, _freq_idx, _p_covmat_l1);

        // wait until the result dma is finished for all iterations > 0
        if (_freq_idx > 0) {
            rt_dma_wait(&_copy);
        }

        // compute whitening
        mrbci_whitening(_p_covmat_l1, _freq_idx, _p_whitened_l1, _p_whitening_workspace_l1);

        // copy back the whitening to l2 memory (we wait for the memcopy to finish in the next iteration or at the end)
        rt_dma_memcpy((unsigned int)_p_whitened_l2_iter,
                      (unsigned int)_p_whitened_l1,
                      sizeof(int32_t) * MRBCI_C * MRBCI_C,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);

        _p_whitened_l2_iter += MRBCI_C * MRBCI_C;

    }

    // free up the input data
    rt_free(RT_ALLOC_CL_DATA, _p_input_l1, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // free the space for the logm input computation
    rt_free(RT_ALLOC_CL_DATA, _p_filter_data_l1, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_covmat_l1, sizeof(int16_t) * MRBCI_C * MRBCI_C);
    rt_free(RT_ALLOC_CL_DATA, _p_whitening_workspace_l1, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // wait for the dma to finish and then free the memory
    rt_dma_wait(&_copy);
    rt_free(RT_ALLOC_CL_DATA, _p_whitened_l1, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    /*
     * Second part: Compute the features for the 18 different frequency bands in parallel
     */

    // allocate the data required for all the computation inside the kernel
    int32_t* _p_whitened_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C * NUM_WORKERS);
    float* _p_logm_workspace_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * _LOGM_WORKSPACE_SIZE * NUM_WORKERS);
    int8_t* _p_logm_out_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN * NUM_WORKERS);
    int8_t* _p_out_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_HALF_DIAG_FEATURES * NUM_WORKERS);

    int _merge = 0;

    for (unsigned int _start_freq_idx = 0; _start_freq_idx < MRBCI_NUM_FREQ; _start_freq_idx += NUM_WORKERS) {

        unsigned int _batch_size = MRBCI_NUM_FREQ - _start_freq_idx;
        if (_batch_size > NUM_WORKERS) {
            _batch_size = NUM_WORKERS;
        }

        // copy the whitened data to l1
        rt_dma_memcpy((unsigned int)(_p_whitened_l2 + _start_freq_idx * MRBCI_C * MRBCI_C),
                      (unsigned int)_p_whitened_l1_th,
                      sizeof(int32_t) * MRBCI_C * MRBCI_C * _batch_size,
                      RT_DMA_DIR_EXT2LOC, _merge, &_copy);

        // prepare the arguments for the kernel
        _mrbci_feature_extraction_kernel_instance_t _args;
        _args.p_whitened = _p_whitened_l1_th;
        _args.p_logm_workspace = _p_logm_workspace_l1_th;
        _args.p_logm_out = _p_logm_out_l1_th;
        _args.p_out = _p_out_l1_th;
        _args.start_freq_idx = _start_freq_idx;

        // wait until the dma is finished
        rt_dma_wait(&_copy);

        // call the cluster
        rt_team_fork(NUM_WORKERS, _mrbci_feature_extraction_kernel, &_args);

        // copy the data back to the result
        rt_dma_memcpy((unsigned int)(p_out + MRBCI_HALF_DIAG_FEATURES * _start_freq_idx),
                      (unsigned int)_p_out_l1_th,
                      sizeof(int8_t) * MRBCI_HALF_DIAG_FEATURES * _batch_size,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);

        // remember that the next memcopy to l1 must be merged
        _merge = 1;

    }

    // wait until the data is copied
    rt_dma_wait(&_copy);

    // free the data required in the kernel
    rt_free(RT_ALLOC_CL_DATA, _p_whitened_l1_th, sizeof(int32_t) * MRBCI_C * MRBCI_C * NUM_WORKERS);
    rt_free(RT_ALLOC_CL_DATA, _p_logm_workspace_l1_th, sizeof(float) * _LOGM_WORKSPACE_SIZE * NUM_WORKERS);
    rt_free(RT_ALLOC_CL_DATA, _p_logm_out_l1_th, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN * NUM_WORKERS);
    rt_free(RT_ALLOC_CL_DATA, _p_out_l1_th, sizeof(int8_t) * MRBCI_HALF_DIAG_FEATURES * NUM_WORKERS);

    rt_free(RT_ALLOC_L2_CL_DATA, _p_whitened_l2, sizeof(int32_t) * MRBCI_NUM_FREQ * MRBCI_C * MRBCI_C);

#else//PARALLEL

    rt_dma_copy_t _copy;

    // allocate output memory in L1
    int8_t* _p_out_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);
    int8_t* _p_out_l1_iter = _p_out_l1;

    // repeat the same procedure for all frequencies
    for (unsigned int freq_idx = 0; freq_idx < MRBCI_NUM_FREQ; freq_idx++) {

        mrbci_extract_features_in_band(p_in, freq_idx, _p_out_l1_iter);
        _p_out_l1_iter += MRBCI_HALF_DIAG_FEATURES;

    }

    // copy back the result
    rt_dma_memcpy((unsigned int)p_out,
                  (unsigned int)_p_out_l1,
                  MRBCI_SVM_NUM_FEATURES,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free the allocated data
    rt_free(RT_ALLOC_CL_DATA, _p_out_l1, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);

#endif//PARALLEL

}

/**
 * @brief compute the features from the input for a given frequency band
 *
 * @warning p_in is not allowed to be in L1, it must reside in L2 memory. p_out however must be in L1.
 * 
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to the entire result vector (of shape MRBCI_HALF_DIAG_NUM_FEATURES) in L1.
 */
void mrbci_extract_features_in_band(const int8_t* p_in,
                                    unsigned int freq_idx,
                                    int8_t* p_out) {

    rt_dma_copy_t _copy;

    // allocate input data
    int8_t* _p_input_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // copy in the data
    rt_dma_memcpy((unsigned int)p_in,
                  (unsigned int)_p_input_l1,
                  sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);

    // allocate the space for the filtered data
    int8_t* _p_filter_data_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // fill the aligned elements with 0 to make sure that the covmat is computed correctly
    unsigned int _last_4_columns = MRBCI_T_ALIGN - 4;
    for (unsigned int _ch = 0; _ch < MRBCI_C; _ch++) {
        *((int32_t*)(_p_filter_data_l1 + MRBCI_T_ALIGN * _ch + _last_4_columns)) = 0;
    }

    // wait until the input data is copied
    rt_dma_wait(&_copy);

    // apply the filtering
    mrbci_filter(_p_input_l1, freq_idx, _p_filter_data_l1);

    // free up the input data
    rt_free(RT_ALLOC_CL_DATA, _p_input_l1, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // allocate the space for the covariance matrix
    int16_t* _p_covmat_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int16_t) * MRBCI_C * MRBCI_C);

    // compute the covariance matrix
    mrbci_covmat(_p_filter_data_l1, freq_idx, _p_covmat_l1);

    // free the filtered data
    rt_free(RT_ALLOC_CL_DATA, _p_filter_data_l1, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // allocate whitening output and workspace for whitening
    int32_t* _p_whitened_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C);
    int32_t* _p_whitening_workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // compute whitening
    mrbci_whitening(_p_covmat_l1, freq_idx, _p_whitened_l1, _p_whitening_workspace_l1);

    // free the covariance matrix data
    rt_free(RT_ALLOC_CL_DATA, _p_covmat_l1, sizeof(int16_t) * MRBCI_C * MRBCI_C);

    // free workspace for whitening
    rt_free(RT_ALLOC_CL_DATA, _p_whitening_workspace_l1, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // allocate result of matrix logarithm and workspace for logm
    int8_t* _p_logm_out_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN);
    float* _p_logm_workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * _LOGM_WORKSPACE_SIZE);

    // compute matrix logarithm
    mrbci_logm(_p_whitened_l1, freq_idx, _p_logm_out_l1, _p_logm_workspace_l1);

    // free whitening output
    rt_free(RT_ALLOC_CL_DATA, _p_whitened_l1, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // free logm workspace
    rt_free(RT_ALLOC_CL_DATA, _p_logm_workspace_l1, sizeof(float) * _LOGM_WORKSPACE_SIZE);

    // compute hald diagonalization
    mrbci_half_diag(_p_logm_out_l1, p_out);

    // free result of matrix logarithm
    rt_free(RT_ALLOC_CL_DATA, _p_logm_out_l1, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN);

}
