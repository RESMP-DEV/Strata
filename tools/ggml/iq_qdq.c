/* IQ3 quantize/dequantize bridge for Strata's ggml tree.
 *
 * type 0: IQ3_S, type 1: IQ3_XXS.  The quantizer and dequantizer are the
 * pinned llama.cpp reference implementations, so this measures the exact
 * block format the engine reads rather than an approximation.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml-common.h"

/* Grid-table initializer from ggml-quants.c (called by ggml.c's
 * ggml_quantize_init in the full build). */
void iq3xs_init_impl(int grid_size);
#include "ggml-quants.h"

/* Minimal ggml helpers used by ggml-quants.c logging and assertion paths.
 * The quantizer itself needs only the block structures above. */
int64_t ggml_blck_size(enum ggml_type type) {
    return (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) ? QK_K : 1;
}

size_t ggml_type_size(enum ggml_type type) {
    if (type == GGML_TYPE_IQ3_S) return sizeof(block_iq3_s);
    if (type == GGML_TYPE_IQ3_XXS) return sizeof(block_iq3_xxs);
    return 0;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    const int64_t block = ggml_blck_size(type);
    return (size_t) (ne / block) * ggml_type_size(type);
}

const char * ggml_type_name(enum ggml_type type) {
    (void) type;
    return "iq_qdq";
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "%s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}

int iq_qdq(const float * src, float * dst, int64_t rows, int64_t cols, int type,
           int64_t * out_bytes) {
    static int initialized_s = 0;
    static int initialized_xxs = 0;
    if (type == 0 && !initialized_s) {
        iq3xs_init_impl(512);
        initialized_s = 1;
    } else if (type == 1 && !initialized_xxs) {
        iq3xs_init_impl(256);
        initialized_xxs = 1;
    }
    int64_t block = 0;
    int64_t block_size = 0;
    switch (type) {
        case 0: block = QK_K;   block_size = sizeof(block_iq3_s);   break;  /* IQ3_S */
        case 1: block = QK_K;   block_size = sizeof(block_iq3_xxs); break;  /* IQ3_XXS */
        case 2: block = QK4_NL; block_size = sizeof(block_iq4_nl);  break;  /* IQ4_NL */
        case 3: block = QK_K;   block_size = sizeof(block_iq4_xs);  break;  /* IQ4_XS */
        default: return -3;
    }
    if (cols % block != 0) {
        return -2;
    }
    const int64_t blocks = cols / block;
    void * tmp = malloc((size_t) (blocks * block_size));
    if (tmp == NULL) {
        return -4;
    }
    for (int64_t row = 0; row < rows; ++row) {
        const float * x = src + row * cols;
        float * y = dst + row * cols;
        if (type == 0) {
            quantize_row_iq3_s_ref(x, (block_iq3_s *) tmp, cols);
            dequantize_row_iq3_s((const block_iq3_s *) tmp, y, cols);
        } else if (type == 1) {
            quantize_row_iq3_xxs_ref(x, (block_iq3_xxs *) tmp, cols);
            dequantize_row_iq3_xxs((const block_iq3_xxs *) tmp, y, cols);
        } else if (type == 2) {
            quantize_row_iq4_nl_ref(x, (block_iq4_nl *) tmp, cols);
            dequantize_row_iq4_nl((const block_iq4_nl *) tmp, y, cols);
        } else {
            quantize_row_iq4_xs_ref(x, (block_iq4_xs *) tmp, cols);
            dequantize_row_iq4_xs((const block_iq4_xs *) tmp, y, cols);
        }
    }
    free(tmp);
    *out_bytes = rows * blocks * block_size;
    return 0;
}

int iq_block_bytes(int type) {
    if (type == 0) return (int) sizeof(block_iq3_s);
    if (type == 1) return (int) sizeof(block_iq3_xxs);
    if (type == 2) return (int) sizeof(block_iq4_nl);
    if (type == 3) return (int) sizeof(block_iq4_xs);
    return -1;
}
