/**
 * @file covmat.c
 * @author Tibor Schneider
 * @date 2020/02/27
 * @brief This file contains the covmat implementation
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

#include "functional.h"
#include "rt/rt_api.h"

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 9
#endif//NUM_WORKERS

typedef struct {
    const int8_t* p_x;
    int32_t rho;
    unsigned int M;
    unsigned int N;
    unsigned int N_align;
    unsigned int y_shift;
    int16_t* p_y;
} _func_covmat_reg_kernel_instance_t;

/**
 * @brief Kernel for doing parallel computation
 */
void _func_covmat_reg_kernel(void* args) {

    unsigned int _core_id = rt_core_id();

    _func_covmat_reg_kernel_instance_t* _args = args;

    const int8_t* p_x = _args->p_x;;
    int32_t rho = _args->rho;
    unsigned int M = _args->M;
    unsigned int N = _args->N;
    unsigned int N_align = _args->N_align;
    unsigned int y_shift = _args->y_shift;
    int16_t* p_y = _args->p_y;

    v4s* _p_x_iter_a;
    v4s* _p_x_iter_b;

    v4s _a0, _a1;
    v4s _b0, _b1;
    int32_t _acc0, _acc1;

    unsigned int num_blk = (M / 4) / 2;
    unsigned int rem_blk = (M / 4) % 2;

    // Stage A: compute the diagonal elements (start from above, such that the cores with low ID have less to do
    for (unsigned int _n = (NUM_WORKERS - 1 - _core_id); _n < N; _n += NUM_WORKERS) {
        _p_x_iter_a = (v4s*)(p_x + _n * M);

        _acc0 = rho; // already add the rho to one of the accumulators
        _acc1 = 0;

        for (unsigned int _m = 0; _m < num_blk; _m++) {
            _a0 = *_p_x_iter_a++;
            _a1 = *_p_x_iter_a++;

            _acc0 = __SUMDOTP4(_a0, _a0, _acc0);
            _acc1 = __SUMDOTP4(_a1, _a1, _acc1);
        }

        if (rem_blk > 0) {
            _a0 = *_p_x_iter_a++;
            _acc0 = __SUMDOTP4(_a0, _a0, _acc0);
        }

        // prepare and store the result
        _acc0 = __ADDROUNDNORM_REG(_acc0, _acc1, y_shift);
        _acc0 = __CLIP(_acc0, 15);
        p_y[_n * (N_align + 1)] = (int16_t)_acc0;

    }

    // Stage B Compute all off-diagonal elements
    // This iterator counts how many places we need to skip until the next element to be computed
    int _skip_positions = _core_id;

    for (unsigned int _i = 1; _i < N; _i++) {
        for (unsigned int _j = 0; _j < _i; _j++) {

            if (_skip_positions > 0) {
                _skip_positions -= 1;
                continue;
            }
            _skip_positions = NUM_WORKERS - 1;

            _p_x_iter_a = (v4s*)(p_x + _i * M);
            _p_x_iter_b = (v4s*)(p_x + _j * M);

            _acc0 = 0;
            _acc1 = 0;

            for (unsigned int _m = 0; _m < num_blk; _m++) {
                _a0 = *_p_x_iter_a++;
                _a1 = *_p_x_iter_a++;
                _b0 = *_p_x_iter_b++;
                _b1 = *_p_x_iter_b++;
                _acc0 = __SUMDOTP4(_a0, _b0, _acc0);
                _acc1 = __SUMDOTP4(_a1, _b1, _acc1);
            }

            if (rem_blk > 0) {
                _a0 = *_p_x_iter_a++;
                _b0 = *_p_x_iter_b++;
                _acc0 = __SUMDOTP4(_a0, _b0, _acc0);
            }

            // prepare and store the result
            _acc0 = __ADDROUNDNORM_REG(_acc0, _acc1, y_shift);
            _acc0 = __CLIP(_acc0, 15);
            p_y[_i * N_align + _j] = (int16_t)_acc0;
            p_y[_j * N_align + _i] = (int16_t)_acc0;
        }
    }
}

#endif //PARALLEL

/**
 * @brief compute the regularized covariance matrix of a matrix X.
 *
 *     Y = X @ X.T
 *
 * @warning p_y must already be allocated on L1
 * @warning M must be 4-aligned! The remaining elements should contain 0s
 *
 * @param p_x Pointer to matrix X of shape [N, M]
 * @param rho Regularization parameter, added to the main diagonal elements
 * @param M Number of columns of matrix X
 * @param N Number of rows of matrix X and dimensionality of matrix Y
 * @param N_align Number of 2-aligned columns of matrix Y
 * @param y_shift Number of bits to shift to the right to store Y
 * @param p_y Pointer to output matrix Y of shape [N, N_align]
 */
void func_covmat_reg(const int8_t* p_x,
                     int32_t rho,
                     unsigned int M,
                     unsigned int N,
                     unsigned int N_align,
                     unsigned int y_shift,
                     int16_t* p_y) {

#ifdef PARALLEL

    _func_covmat_reg_kernel_instance_t _args;
    _args.p_x = p_x;
    _args.rho = rho;
    _args.M = M;
    _args.N = N;
    _args.N_align = N_align;
    _args.y_shift = y_shift;
    _args.p_y = p_y;

    rt_team_fork(NUM_WORKERS, _func_covmat_reg_kernel, &_args);

#else //PARALLEL

    v4s* _p_x_iter_a;
    v4s* _p_x_iter_b;

    v4s _a0, _a1;
    v4s _b0, _b1;
    int32_t _acc0, _acc1;

    if (M % 4) {
        printf("Error: M must be aligned");
        return;
    }

    unsigned int num_blk = (M / 4) / 2;
    unsigned int rem_blk = (M / 4) % 2;

    // Stage A: compute the diagonal elements
    for (unsigned int _n = 0; _n < N; _n++) {
        _p_x_iter_a = (v4s*)(p_x + _n * M);

        _acc0 = rho; // already add the rho to one of the accumulators
        _acc1 = 0;

        for (unsigned int _m = 0; _m < num_blk; _m++) {
            _a0 = *_p_x_iter_a++;
            _a1 = *_p_x_iter_a++;

            _acc0 = __SUMDOTP4(_a0, _a0, _acc0);
            _acc1 = __SUMDOTP4(_a1, _a1, _acc1);
        }

        if (rem_blk > 0) {
            _a0 = *_p_x_iter_a++;
            _acc0 = __SUMDOTP4(_a0, _a0, _acc0);
        }

        // prepare and store the result
        _acc0 = __ADDROUNDNORM_REG(_acc0, _acc1, y_shift);
        _acc0 = __CLIP(_acc0, 15);
        p_y[_n * (N_align + 1)] = (int16_t)_acc0;

    }

    // Stage B Compute all off-diagonal elements
    for (unsigned int _i = 1; _i < N; _i++) {
        for (unsigned int _j = 0; _j < _i; _j++) {

            _p_x_iter_a = (v4s*)(p_x + _i * M);
            _p_x_iter_b = (v4s*)(p_x + _j * M);

            _acc0 = 0;
            _acc1 = 0;

            for (unsigned int _m = 0; _m < num_blk; _m++) {
                _a0 = *_p_x_iter_a++;
                _a1 = *_p_x_iter_a++;
                _b0 = *_p_x_iter_b++;
                _b1 = *_p_x_iter_b++;
                _acc0 = __SUMDOTP4(_a0, _b0, _acc0);
                _acc1 = __SUMDOTP4(_a1, _b1, _acc1);
            }

            if (rem_blk > 0) {
                _a0 = *_p_x_iter_a++;
                _b0 = *_p_x_iter_b++;
                _acc0 = __SUMDOTP4(_a0, _b0, _acc0);
            }

            // prepare and store the result
            _acc0 = __ADDROUNDNORM_REG(_acc0, _acc1, y_shift);
            _acc0 = __CLIP(_acc0, 15);
            p_y[_i * N_align + _j] = (int16_t)_acc0;
            p_y[_j * N_align + _i] = (int16_t)_acc0;
        }
    }

#endif //PARALLEL

}
