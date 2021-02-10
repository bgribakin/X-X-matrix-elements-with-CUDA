#pragma once

#include <cstdio>
#include <stdlib.h>
#include <cstring>

void printFile(const char* filename, FILE* filePointer);
void fileSkipLines(FILE* F, int linesSkipped);
int get_val_from_line(char* filename, const char* type, int line, int* int_val, double* double_val);