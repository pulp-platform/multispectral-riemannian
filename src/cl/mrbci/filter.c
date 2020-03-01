/**
 * @file filter.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for the filtering
 */

#include "mrbci.h"
#include "../func/functional.h"

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif//NUM_WORKERS

typedef struct {
    const int8_t * p_in;
    unsigned int freq_idx;
    int8_t * p_out;
} _mrbci_filter_kernel_instance_t;

/**
 * @brief Kernel for doing parallel computation
 */
void _mrbci_filter_kernel(void* args) {

    unsigned int _core_id = rt_core_id();

    _mrbci_filter_kernel_instance_t* _args = args;

    const int8_t * _p_in = _args->p_in;
    unsigned int _freq_idx = _args->freq_idx;
    int8_t * _p_out = _args->p_out;

    // setup channels
    unsigned int _ch = _core_id;

    //setup iterators
    const int8_t * _p_in_iter = _p_in + _ch * MRBCI_T_ALIGN;
    int8_t * _p_out_iter = _p_out + _ch * MRBCI_T_ALIGN;

    while (_ch < MRBCI_C) {

        func_sos_filt_2S(_p_in_iter,
                         MRBCI_T,
                         &(mrbci_filter_params[_freq_idx]),
                         _p_out_iter);

        // go to the next channel
        _ch += NUM_WORKERS;
        _p_in_iter += NUM_WORKERS * MRBCI_T_ALIGN;
        _p_out_iter += NUM_WORKERS * MRBCI_T_ALIGN;

    }

    rt_team_barrier();

}

#endif //PARALLEL

/**
 * @brief Apply the FIR filter for a given frequency
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input data of shape [C, T], aligned to [C, T_ALIGN]
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, T], aligned to [C, T_ALIGN]
 */
void mrbci_filter(const int8_t* p_in,
                  unsigned int freq_idx,
                  int8_t* p_out) {

#ifdef PARALLEL

    _mrbci_filter_kernel_instance_t _args;
    _args.p_in = p_in;
    _args.freq_idx = freq_idx;
    _args.p_out = p_out;

    rt_team_fork(NUM_WORKERS, _mrbci_filter_kernel, &_args);

#else //PARALLEL

    for (int _ch = 0; _ch < MRBCI_C; _ch++) {
        func_sos_filt_2S(p_in + _ch * MRBCI_T_ALIGN,
                         MRBCI_T,
                         &(mrbci_filter_params[freq_idx]),
                         p_out + _ch * MRBCI_T_ALIGN);
    }

#endif //PARALLEL

}
