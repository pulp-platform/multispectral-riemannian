#include "math.h"

void householder_tridiagonal(float* p_a, float* p_q, unsigned int N, float* p_workspace);
float vnorm_f(const float* p_a, unsigned int N, unsigned int stride);
void vcovmat_f(const float* p_a, unsigned int N, float* p_y);
void matmul_f(const float* p_a, const float* p_b, unsigned int M, unsigned int N, unsigned int O, float* p_y);
void copy_mat(float* p_src, float* p_dst, unsigned int N, unsigned int M, unsigned int stride_src, unsigned int stride_dst);

void householder_tridiagonal(float* p_a,
                             float* p_q,
                             unsigned int N,
                             float* p_workspace) {

    float* _p_v = p_workspace;
    float* _p_h = p_workspace + N;
    float* _p_tmp = p_workspace + N * (N + 1);

    float* _p_a = p_a;
    float* _p_q = p_q;

    float* _p_a_iter = _p_a;
    float* _p_v_iter;
    float* _p_h_iter;

    float* _p_swap;

    // Start with the iterations
    for (unsigned int _k = 0; _k < N - 2; _k++) {

        // pointer to A[k,k+1]
        _p_a_iter = _p_a + _k * (N + 1) + 1;

        // compute the scale of the row right of the current diagonal element
        float _scale = vnorm_f(_p_a_iter, (N - _k - 1), 1);
        if (_scale == 0.f) {
            continue;
        }

        float _val = *(_p_a_iter++); // _p_a_iter now points to A[k,k+2]
        float _sign = copysignf(1.0f, _val);
        float _scaled_val = _sign * _val / _scale;
        float _z = (1.f + _scaled_val) * 0.5f;
        float _sqrtz = sqrtf(_z);
        float _vec_scale = 1.f / (2.f * _scale * _sqrtz);

        // generate vector _p_v
        _p_v_iter = _p_v;
        for (unsigned int _i = 0; _i < _k + 1; _i++) {
            *_p_v_iter++ = 0.f;
        }
        *_p_v_iter++ = _sqrtz;
        for (unsigned int _i = _k + 2; _i < N; _i++) {
            // read the element of A and multiply with the sign of _val
            float _tmp_val = *_p_a_iter++ * _sign;
            // write the vector
            *_p_v_iter++ = _tmp_val * _vec_scale;
        }

        // Generate the rotation matrix H
        vcovmat_f(_p_v, N, _p_h);
        _p_h_iter = _p_h;
        for (unsigned int _i = 0; _i < N; _i++) {
            for (unsigned int _j = 0; _j < N; _j++) {
                if (_i == _j) {
                    *_p_h_iter = 1 - 2.f * (*_p_h_iter);
                } else {
                    *_p_h_iter = -2.f * (*_p_h_iter);
                }
                _p_h_iter++;
            }
        }

        // transform the matrices with H
        matmul_f(_p_a, _p_h, N, N, N, _p_tmp);
        matmul_f(_p_h, _p_tmp, N, N, N, _p_a);

        matmul_f(_p_q, _p_h, N, N, N, _p_tmp);

        // swap pointers tmp and q
        _p_swap = _p_tmp;
        _p_tmp = _p_q;
        _p_q = _p_swap;

    }

    /*
     * We did the rotation (N-2) times in total. Thus, if (N-2) 2 3 == 0, the data is already at the
     * correct position. 
     */
    float* _o_tmp = p_workspace + N * (N + 1); // reconstruct the original tmp matrix
    if (_p_q != p_q) {
        copy_mat(_o_tmp, p_q, N, N, N, N);
    }

}

float vnorm_f(const float* p_a,
                     unsigned int N,
                     unsigned int stride) {

    /*
     * Loop Unrolling does not improve things because we cannot do load post increment!
     */

    float _acc = 0;
    float _val;

    for (unsigned int _i = 0; _i < N; _i++) {

        _val = *p_a;
        p_a += stride;
        _acc += _val * _val;

    }

    _acc = sqrtf(_acc);

    return _acc;

}

void vcovmat_f(const float* p_a,
               unsigned int N,
               float* p_y) {

    const float* _p_a_iter1 = p_a;
    const float* _p_a_iter2;

    float _val_a;

    float* _p_y_iter1;
    float* _p_y_iter2;
    float _val_y;

    for (unsigned int _i = 0; _i < N; _i++) {

        // prepare the iterators
        _p_y_iter1 = p_y + _i * (N + 1);
        _p_y_iter2 = _p_y_iter1 + N;
        _p_a_iter2 = _p_a_iter1 + 1;

        // load the first value, which is constant for this entire cycle
        _val_a = *_p_a_iter1++;

        // compute the diagonal element and store back
        *_p_y_iter1++ = _val_a * _val_a;

        // compute the remaining elements
        for (unsigned int _j = _i + 1; _j < N; _j++) {
            _val_y = _val_a * (*_p_a_iter2++);
            *_p_y_iter1 = _val_y;
            *_p_y_iter2 = _val_y;
            _p_y_iter1 += 1;
            _p_y_iter2 += N;
        }
    }

}

void matmul_f(const float* p_a,
              const float* p_b,
              unsigned int M,
              unsigned int N,
              unsigned int O,
              float* p_y) {

    /*
     * Compute N elements of the output matrix at a time, reducing the number of required elements 
     * to load
     */

    // use iterators
    const float* _p_a_iter = p_a;
    const float* _p_b_iter = p_b;
    const float* _p_a_iter_comp;
    const float* _p_b_iter_comp;
    float* _p_y_iter = p_y;

    float _acc;

    // loop over every output element
    for (unsigned int _m = 0; _m < M; _m++) {
        for (unsigned int _o = 0; _o < O; _o++) {

            _acc = 0;
            _p_a_iter_comp = _p_a_iter;
            _p_b_iter_comp = _p_b_iter;

            for (unsigned int _n = 0; _n < N; _n++) {
                _acc += (*_p_a_iter_comp) * (*_p_b_iter_comp);

                _p_a_iter_comp++;
                _p_b_iter_comp += O;
            }

            *(_p_y_iter++) = _acc;

            _p_b_iter++;
        }

        _p_a_iter += N;
        _p_b_iter = p_b;
    }

}

void copy_mat(float* p_src,
              float* p_dst,
              unsigned int N,
              unsigned int M,
              unsigned int stride_src,
              unsigned int stride_dst) {

    float* _p_src_iter = p_src;
    float* _p_dst_iter = p_dst;

    unsigned int _newline_src = stride_src - M;
    unsigned int _newline_dst = stride_dst - M;

    for (unsigned int _n = 0; _n < N; _n++) {
        for (unsigned int _m = 0; _m < M; _m++) {
            *_p_dst_iter++ = *_p_src_iter++;
        }

        _p_src_iter += _newline_src;
        _p_dst_iter += _newline_dst;
    }

}
