#include "common.h"

// Scan / all prefix operation is as follows:
// given a series of numbers, associative operator, and identity I,
// for array [a0, a1, a2, ..., an-1], we get output after scan:
// [I, a0, (a0 OPR a1), (a0 OPR a1 OPR a2), (...)]
// This is a type of exclusive scan (each element is the sum of all elements upto it)
