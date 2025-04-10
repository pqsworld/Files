#ifndef __BMP_H__
#define __BMP_H__
int LoadBmp(char filename[], unsigned char* ucImage, int* h, int* w);
int WriteBmp(char filename[], unsigned char* ucImage, int h, int w);

#endif //结束第一个#ifndef
