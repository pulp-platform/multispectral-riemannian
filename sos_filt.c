#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

int quant_sos_filt_df1(const int64_t* x,
                       const int64_t* a,
                       const int64_t* b,
                       const int64_t* a_shift,
                       const int64_t* b_shift,
                       int N_prime,
                       int M,
                       int intermediate_bits,
                       int64_t* y) {
    
    int clip_range_neg = -(1 << (intermediate_bits - 1));
    int clip_range_pos = (1 << (intermediate_bits - 1)) - 1;

    int64_t** regs = malloc(M * sizeof(int64_t*));
    for (int i = 0; i < M; i++) {
        regs[i] = malloc(3 * sizeof(int64_t));
        for (int j = 0; j < 3; j++) {
            regs[i][j] = 0;
        }
    }

    // main body
    for (int k = 0; k < N_prime; k++) {

        // prepare input registers
        regs[0][2] = regs[0][1];
        regs[0][1] = regs[0][0];
        regs[0][0] = x[k];

        // sum over all sections
        for (int m = 1; m < M; m++) {
            regs[m][2] = regs[m][1];
            regs[m][1] = regs[m][0];

            int64_t acc = 0;
            int64_t tmp = 0;
            tmp = (regs[m - 1][0] * b[m * 3 + 0]);
            tmp += (regs[m - 1][1] * b[m * 3 + 1]);
            tmp += (regs[m - 1][2] * b[m * 3 + 2]);
            //acc = (tmp + (1 << (b_shift[m] - 1))) >> b_shift[m];
            acc = tmp >> b_shift[m];

            tmp = (regs[m][1] * a[m * 3 + 1]);
            tmp += (regs[m][2] * a[m * 3 + 2]);
            //acc -= (tmp + (1 << (a_shift[m] - 1))) >> a_shift[m];
            acc -= tmp >> a_shift[m];

            // handle overflow
            if (acc < clip_range_neg || acc > clip_range_pos) {
                printf("overflow with %ld\n", acc);
                return 1;
            }

            regs[m][0] = acc;

        }

        // store the output
        y[k] = regs[M - 1][0];

    }


    return 0;

}
