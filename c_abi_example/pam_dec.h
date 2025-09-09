#include <stdint.h>

// Simple PAM decoder for SSIMULACRA2 testing

// Image struct returned by PAM decoder
typedef struct Image {
    uint32_t width;
    uint32_t height;
    uint8_t channels;
    uint8_t* data;
} Image;

// Load & decode a PAM image given a path
Image *const loadPAM(const char* path);
