#include "fileFunctions.h"

// prints our file with filenames
void printFile(const char* filename, FILE* filePointer) {
    int bufferLength = 255;
    char* buffer = new char[bufferLength];

    printf("'%s':\n", filename);
    while (fgets(buffer, bufferLength, filePointer))
        printf("\t%s", buffer);
    printf("\n\n");
    rewind(filePointer);

    delete[] buffer;
}

void fileSkipLines(FILE* F, int linesSkipped) {
    for (int i = 0; i < linesSkipped; i++)
        fscanf(F, "%*[^\n]\n");
}

int get_val_from_line(char* filename, const char* type, int line, int* int_val, double* double_val) {
    FILE* filePointer;
    int bufferLength = 255;
    char* buffer = new char[bufferLength];

    filePointer = fopen(filename, "r");
    if (filePointer == NULL) {
        printf("\t Error: file does not exist.\n");
        return 0;
    }
    else {
        fileSkipLines(filePointer, line - 1);
        fgets(buffer, bufferLength, filePointer);
//        printf("\nbuffer: %s", buffer);
        if (strcmp(type, "double") == 0) {
            sscanf(buffer, "%*[^0123456789]%lf\n", double_val);
//            printf("val = %f\n", *double_val);
        }
        else if (strcmp(type, "int") == 0) {
            sscanf(buffer, "%*[^0123456789]%d\n", int_val);
//            printf("val = %d\n", *int_val);
        }

        fclose(filePointer);
        return 1;
    }

    delete[] buffer;
}