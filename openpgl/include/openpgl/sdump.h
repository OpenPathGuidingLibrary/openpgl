#pragma once

#include <stdint.h>

struct SDumpLeaf {

};

struct SDumpTree {
    float split;
    uint8_t axis;
    SDumpTree*left, *right;

};

struct SDump {
    SDumpTree *sur;
    SDumpTree *vol;
};