#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "pam_dec.h"
#include "../zig-out/include/ssimu2.h"

static void freeImage(Image* const img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <reference.pam> <distorted.pam>\n", argv[0]);
        return 1;
    }

    Image* const ref = loadPAM(argv[1]);
    if (!ref) {
        printf("Failed to load reference image: %s\n", argv[1]);
        return 1;
    }

    Image* const dist = loadPAM(argv[2]);
    if (!dist) {
        printf("Failed to load distorted image: %s\n", argv[2]);
        freeImage(ref);
        return 1;
    }

    if (ref->width != dist->width || ref->height != dist->height || ref->channels != dist->channels) {
        printf("Images must have the same dimensions and number of channels\n");
        freeImage(ref);
        freeImage(dist);
        return 1;
    }

    double score;
    const int err = ssimulacra2_score(ref->data, dist->data, ref->width, ref->height, ref->channels, &score);

    if (err == SSIMU2_OK)
        printf("SSIMULACRA2 Score: %.6f\n", score);
    else if (err == SSIMU2_INVALID_CHANNELS)
        printf("Error: Invalid number of channels (must be 3 or 4)\n");
    else if (err == SSIMU2_OUT_OF_MEMORY)
        printf("Error: Out of memory\n");
    else
        printf("Unknown error: %d\n", err);

    freeImage(ref);
    freeImage(dist);
    return 0;
}
