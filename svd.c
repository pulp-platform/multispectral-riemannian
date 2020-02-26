#include "math.h"

#define _GIVENS_SAVE_MIN 1.0e-10
#define _SQRT2 1.41421353816986083984f
#define _SVD_PRECISION 1.e-4f
#define _EPSILON 1.1920928955078125e-7f

typedef struct {
    float cs;
    float sn;
} givens_rotation_t;

typedef struct {
    float ev1;
    float ev2;
    float cs;
    float sn;
} evd_2x2_t;

void svd_sym_tridiag(float* p_main_diag, float* p_off_diag, float* p_q, unsigned int N, unsigned int stride, unsigned int current_pos);
void householder_tridiagonal(float* p_a, float* p_q, unsigned int N, float* p_workspace);
float vnorm_f(const float* p_a, unsigned int N, unsigned int stride);
void vcovmat_f(const float* p_a, unsigned int N, float* p_y);
void matmul_f(const float* p_a, const float* p_b, unsigned int M, unsigned int N, unsigned int O, float* p_y);
void copy_mat(float* p_src, float* p_dst, unsigned int N, unsigned int M, unsigned int stride_src, unsigned int stride_dst);
void apply_givens_rotation_f(float* p_a, givens_rotation_t rot, unsigned int k, unsigned int N);
givens_rotation_t givens_rotation(float a, float b);
givens_rotation_t givens_rotation_diag(float a, float b, float c);
evd_2x2_t evd_2x2(float a, float b, float c);

void svd_sym_tridiag(float* p_main_diag,
                     float* p_off_diag,
                     float* p_q,
                     unsigned int N,
                     unsigned int stride,
                     unsigned int current_pos) {

    // m: main iterator
    int _m = N - 1;

    // special cases to break out of recurrence
    if (N == 1) {
        return;
    }

    if (N == 2) {
        // Compute the diagonalization of the 2x2 matrix
        evd_2x2_t _evd = evd_2x2(*p_main_diag, *p_off_diag, *(p_main_diag + 1));

        // write the eigenvalues into the main diag
        *(p_main_diag + 0) = _evd.ev1;

        // apply the rotation
        givens_rotation_t _rot = {_evd.cs, _evd.sn};
        apply_givens_rotation_f(p_q, _rot, current_pos, stride);

        // this instruction needs to be here, because GCC fails if the two instructions are one after the other
        // If GCC does not like the code like this, fall back to O2
        *(p_main_diag + 1) = _evd.ev2;

        return;
    }

    // If N >= 3, do the normal QR decomposition
    while (_m > 0) {

        // check if the matrix left to be transformed has off diagonal elements which are zero
        for (int _k = 0; _k < _m - 1; _k++) {
            if (fabs(p_off_diag[_k]) < _EPSILON) {
                // Divide and Conquer! decompose the matrix
                svd_sym_tridiag(p_main_diag,
                                p_off_diag,
                                p_q,
                                _k + 1, stride, 0);

                svd_sym_tridiag(p_main_diag + _k + 1,
                                p_off_diag + _k + 1,
                                p_q,
                                _m - _k, stride, _k + 1);

                // return
                return;
            }
        }

        // do wilkinson shift
        float _shift;
        float _d = (p_main_diag[_m - 1] - p_main_diag[_m]) * 0.5f;
        if (_d == 0) {
            _shift = p_main_diag[_m] - fabs(p_off_diag[_m - 1]);
        } else {
            float _off_diag_pow2 = p_off_diag[_m - 1] * p_off_diag[_m - 1];
            float _tmp = sqrtf(_d * _d + _off_diag_pow2);
            _tmp = copysign(_tmp, _d);
            _shift = p_main_diag[_m] - _off_diag_pow2 / (_d + _tmp);
        }

        // start the implicit QR step
        float _x = p_main_diag[0] - _shift;
        float _y = p_off_diag[0];

        for (int _k = 0; _k < _m; _k++) {

            // determine the givens rotation
            givens_rotation_t _rot;
            if (_m > 1) {
                _rot = givens_rotation(_x, _y);
            } else {
                _rot = givens_rotation_diag(p_main_diag[0], p_off_diag[0], p_main_diag[1]);
            }

            // compute some values
            float _w = _rot.cs * _x - _rot.sn * _y;
            float _d = p_main_diag[_k] - p_main_diag[_k + 1];
            float _z = (2 * _rot.cs * p_off_diag[_k] + _d * _rot.sn) * _rot.sn;

            // do the step on the main and off diagonal
            p_main_diag[_k] = p_main_diag[_k] - _z;
            p_main_diag[_k + 1] = p_main_diag[_k + 1] + _z;
            p_off_diag[_k] = p_off_diag[_k] * (_rot.cs * _rot.cs - _rot.sn * _rot.sn) + _d * _rot.cs * _rot.sn;
            if (_k > 0) {
                p_off_diag[_k - 1] = _w;
            }

            // update x and y
            _x = p_off_diag[_k];
            if (_k < _m - 1) {
                _y = -_rot.sn * p_off_diag[_k + 1];
                p_off_diag[_k + 1] = _rot.cs * p_off_diag[_k + 1];
            }

            // update the eigenvectors
            _rot.sn = -_rot.sn; // change the sign for the rotation because the sine is defined differently here!
            apply_givens_rotation_f(p_q, _rot, current_pos + _k, stride);

        }

        // check for convergence
        if (fabs(p_off_diag[_m - 1]) < _SVD_PRECISION * (fabs(p_main_diag[_m - 1]) + fabs(p_main_diag[_m]))) {
            _m -= 1;
        }

    }

    return;

}

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

void apply_givens_rotation_f(float* p_a,
                             givens_rotation_t rot,
                             unsigned int k,
                             unsigned int N) {

    float* _p_a_iter = p_a + k;

    float _val_a, _val_b;
    float _res_a, _res_b;

    // loop over all rows
    for (unsigned int _i = 0; _i < N; _i++) {
        // load the two values
        _val_a = *(_p_a_iter + 0);
        _val_b = *(_p_a_iter + 1);

        // compute the new values
        _res_a = _val_a * rot.cs + _val_b * rot.sn;
        _res_b = _val_b * rot.cs - _val_a * rot.sn;

        // store back
        *(_p_a_iter + 0) = _res_a;
        *(_p_a_iter + 1) = _res_b;

        // go to the next line
        _p_a_iter += N;
    }
}

givens_rotation_t givens_rotation(float a,
                                  float b) {

    givens_rotation_t res;

    if (b == 0.f) {
        res.cs = 1.f;
        res.sn = 0.f;
    } else if (a == 0.f) {
        res.cs = 0.f;
        res.sn = copysignf(1.f, b);
    } else {
        float scale = fmax(fabs(a), fabs(b));
        if (scale < _GIVENS_SAVE_MIN) {
            a = a / scale;
            b = b / scale;
        }
        float r = sqrtf(a * a + b * b);
        res.cs = a / r;
        res.sn = -b / r;
    }
    return res;
}

givens_rotation_t givens_rotation_diag(float a,
                                       float b,
                                       float c) {

    float sm = a + c;
    float df = a - c;
    float tb = b + b;
    float adf = fabs(df);
    float ab = fabs(tb);

    float rt;
    int sgn1;
    int sgn2;
    float cs;
    float ct;
    float tn;

    givens_rotation_t res;

    if (adf > ab) {
        rt = adf * sqrtf(1.f + (ab / adf) * (ab / adf));
    } else if (adf < ab) {
        rt = ab * sqrtf(1.f + (adf / ab) * (adf / ab));
    } else {
        rt = ab * _SQRT2;
    }

    if (sm < 0.f) {
        sgn1 = -1;
    } else {
        sgn1 = 1;
    }

    if (df >= 0.f) {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }

    if (fabs(cs) > ab) {
        ct = -tb / cs;
        res.sn = 1.f / sqrtf(1.f + ct * ct);
        res.cs = res.sn * ct;
    } else {
        if (ab == 0.f) {
            res.cs = 1.f;
            res.sn = 0.f;
        } else {
            tn = -cs / tb;
            res.cs = 1.f / sqrtf(1.f + tn * tn);
            res.sn = tn * res.cs;
        }
    }

    if (sgn1 == sgn2) {
        tn = res.cs;
        res.cs = -res.sn;
        res.sn = -tn;
    } else {
        res.sn = -res.sn;
    }

    return res;
}

evd_2x2_t evd_2x2(float a,
                  float b,
                  float c) {

    float sm = a + c;
    float df = a - c;
    float tb = b + b;
    float adf = fabs(df);
    float ab = fabs(tb);

    float acmx;
    float acmn;
    float rt;
    int sgn1;
    int sgn2;
    float cs;
    float ct;
    float tn;

    evd_2x2_t res;

    if (fabs(a) > fabs(c)) {
        acmx = a;
        acmn = c;
    } else {
        acmx = c;
        acmn = a;
    }

    if (adf > ab) {
        rt = adf * sqrtf(1.f + (ab / adf) * (ab / adf));
    } else if (adf < ab) {
        rt = ab * sqrtf(1.f + (adf / ab) * (adf / ab));
    } else {
        rt = ab * _SQRT2;
    }

    if (sm < 0.f) {
        res.ev1 = 0.5f * (sm - rt);
        sgn1 = -1;
        res.ev2 = (acmx / res.ev1) * acmn - (b / res.ev1) * b;
    } else if (sm > 0.f) {
        res.ev1 = 0.5f * (sm + rt);
        sgn1 = 1;
        res.ev2 = (acmx / res.ev1) * acmn - (b / res.ev1) * b;
    } else {
        res.ev1 = 0.5f * rt;
        res.ev2 = -0.5f * rt;
        sgn1 = 1;
    }

    if (df >= 0.f) {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }

    if (fabs(cs) > ab) {
        ct = -tb / cs;
        res.sn = 1.f / sqrtf(1.f + ct * ct);
        res.cs = res.sn * ct;
    } else {
        if (ab == 0.f) {
            res.cs = 1.f;
            res.sn = 0.f;
        } else {
            tn = -cs / tb;
            res.cs = 1.f / sqrtf(1.f + tn * tn);
            res.sn = tn * res.cs;
        }
    }

    if (sgn1 == sgn2) {
        tn = res.cs;
        res.cs = -res.sn;
        res.sn = tn;
    }

    return res;
}
