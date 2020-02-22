/**
 * @file linalg.h
 * @author Tibor Schneider
 * @date 2020/02/21
 * @brief This file contains the definitions for linalg functions
 */

#ifndef __CL_LINALG_H__
#define __CL_LINALG_H__

/**
 * @struct linalg_givens_rotation_t
 * @brief structure containing the sine and cosine for a givens rotation
 *
 * @var linalg_givens_rotation_t::cs cosine for the rotation
 * @var linalg_givens_rotation_t::cs sine for the rotation
 */
typedef struct {
    float cs;
    float sn;
} linalg_givens_rotation_t;

/**
 * @brief Computes the givens rotation coefficients cosine and sine
 *
 * | c -s | | a | = | r |
 * | s  c | | b | = | 0 |
 *
 * @param a first value
 * @param b second value
 * @returns linalg_givens_rotation_t structure
 */
linalg_givens_rotation_t linalg_givens_rotation(float a,
                                                float b);

#endif //__CL_LINALG_H__
