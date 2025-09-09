#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "pam_dec.h"

Image *const loadPAM(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;

    fseek(file, 0, SEEK_END);
    const size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);

    uint8_t* const buf = malloc(size);
    if (!buf) {
        fclose(file);
        return NULL;
    }
    fread(buf, 1, size, file);
    fclose(file);

    if (size < 3 || memcmp(buf, "P7", 2)) {
        free(buf);
        return NULL;
    }

    const char* header_end = strstr((char*)buf, "ENDHDR\n");
    if (!header_end) {
        free(buf);
        return NULL;
    }
    const size_t header_len = header_end - (char*)buf + 7;

    char* header = malloc(header_len + 1);
    if (!header) {
        free(buf);
        return NULL;
    }
    memcpy(header, buf, header_len);
    header[header_len] = '\0';

    uint32_t width, height, depth, maxval;
    const char *line = strtok(header, "\n");
    while (line) {
        if (line[0] == '#') {
            line = strtok(NULL, "\n");
            continue;
        }
        if (strstr(line, "WIDTH"))
            sscanf(line, "WIDTH %u", &width);
        else if (strstr(line, "HEIGHT"))
            sscanf(line, "HEIGHT %u", &height);
        else if (strstr(line, "DEPTH"))
            sscanf(line, "DEPTH %u", &depth);
        else if (strstr(line, "MAXVAL"))
            sscanf(line, "MAXVAL %u", &maxval);
        line = strtok(NULL, "\n");
    }
    free(header);

    if (!width || !height || (depth != 1 && depth != 3) || maxval != 255) {
        free(buf);
        return NULL;
    }

    const size_t original_data_size = (size_t)width * height * depth;
    if (header_len + original_data_size > size) {
        free(buf);
        return NULL;
    }

    uint8_t* original_data = malloc(original_data_size);
    if (!original_data) {
        free(buf);
        return NULL;
    }
    memcpy(original_data, buf + header_len, original_data_size);
    free(buf);

    const size_t data_size = (size_t)width * height * 3;
    uint8_t *const data = malloc(data_size);
    if (!data) {
        free(original_data);
        return NULL;
    }

    if (depth == 1) {
        for (size_t i = 0; i < (size_t)width * height; i++) {
            uint8_t gray = original_data[i];
            data[i * 3 + 0] = gray;
            data[i * 3 + 1] = gray;
            data[i * 3 + 2] = gray;
        }
    } else
        memcpy(data, original_data, data_size);
    free(original_data);

    Image* const img = malloc(sizeof(Image));
    if (!img) {
        free(data);
        return NULL;
    }
    img->width = width;
    img->height = height;
    img->channels = 3;
    img->data = data;
    return img;
}
