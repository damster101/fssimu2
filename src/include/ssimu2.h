#ifndef SSIMULACRA2_H
#define SSIMULACRA2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define SSIMU2_OK 0
#define SSIMU2_INVALID_CHANNELS 1
#define SSIMU2_OUT_OF_MEMORY 2

// Compute a SSIMULACRA2 score
// The caller must ensure that the reference and distorted buffers
// are at least (width * height * channels) bytes long. If not,
// could lead to UB in ReleaseFast
int ssimulacra2_score(
    const uint8_t *reference,
    const uint8_t *distorted,
    const unsigned width,
    const unsigned height,
    const unsigned channels,
    const double *out_score
);

#ifdef __cplusplus
}
#endif

#endif // SSIMULACRA2_H
