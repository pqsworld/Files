//#include "stdafx.h"
#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include "bmp.h"

/*定义*/

int LoadBmp(char filename[], unsigned char* ucImage, int* h, int* w)
{
    printf("filename:%s", filename);
    FILE* fp;
    int i, j, l;
    int h1, w1;
    unsigned char c;
    unsigned int offset;

    fp = fopen(filename, "rb");

    if (NULL == fp) {
        printf("open fail");

        return -1;
    }

    fseek(fp, 10, SEEK_SET);
    fread(&offset, 1, 4, fp);

    fseek(fp, 18, SEEK_SET);
    fread(&w1, 1, 4, fp);
    fread(&h1, 1, 4, fp);

    printf("w1:%d", w1);
    printf("h1:%d", h1);
    *w = w1;
    *h = h1;

    fseek(fp, offset, SEEK_SET);

    l = (*w + 3) / 4 * 4;

    for (i = 0; i < *h; i++)
        for (j = 0; j < l; j++) {
            c = fgetc(fp);

            if (j < *w) {
                ucImage[(*h - 1 - i) * (*w) + j] = c;
            }
        }

    fclose(fp);

    return (*h) * (*w);
}

int WriteBmp(char filename[], unsigned char* ucImage, int h, int w)
{
    FILE* fp;
    int i, j, l, offset, size;
    unsigned char* p;
    unsigned char bmpheader[54] = {
        //注意：小端序
        // BitmapFileHeader
        0x42, 0x4d,             // bfType            BMP类型，一般为'BM'
        0x00, 0x00, 0x00, 0x00, // bfSize            BMP图像（包括头部）的大小
        0x00, 0x00,             // bfReserved1       保留项1，固定为0
        0x00, 0x00,             // bfReserved2       保留项2，固定为0
        0x36, 0x04, 0x00, 0x00, // bfOffBits         位图区偏移的比特数，灰度图一般为14 + 40 + 1024 = 1078（0x000436）

        // BitmapInfoHeader
        0x28, 0x00, 0x00, 0x00, // biSize            信息头部大小，固定为40
        0x00, 0x00, 0x00, 0x00, // biWidth           图像的宽
        0x00, 0x00, 0x00, 0x00, // biHeight          图像的高
        0x01, 0x00,             // biPlanes          设备等级，固定为1
        0x08, 0x00,             // biBitCount        BMP单个像素的比特数，灰度图固定为8
        0x00, 0x00, 0x00, 0x00, // biCompression     BMP压缩类型，一般为0（无压缩）
        0x00, 0x00, 0x00, 0x00, // biSizeImage       BMP位图区大小（ biCompression = 0时本项可以为0）
        0xc4, 0x0e, 0x00, 0x00, // biXPelsPerMeter   水平分辨率
        0xc4, 0x0e, 0x00, 0x00, // biYPelsPerMeter   垂直分辨率
        0x00, 0x00, 0x00, 0x00, // biClrUsed         使用的颜色数量，一般为0（都使用）
        0x00, 0x00, 0x00, 0x00  // biClrImportant    重要颜色的数量，一般为0（都重要）
    };

    fp = fopen(filename, "wb");

    if (NULL == fp) {
        return -1;
    }

    offset = 0x000436;
    l = (w + 3) / 4 * 4;
    size = l * h + offset;

    fwrite(bmpheader, 1, 2, fp);
    fwrite(&size, 1, 4, fp);
    fwrite(bmpheader + 6, 1, 12, fp);
    fwrite(&w, 1, 4, fp);
    fwrite(&h, 1, 4, fp);
    fwrite(bmpheader + 26, 1, 28, fp);

    for (i = 0; i < 256; i++) {
        fputc(i, fp);
        fputc(i, fp);
        fputc(i, fp);
        fputc(0, fp);
    }

    p = ucImage;

    for (i = h - 1; i >= 0; i--)
        for (j = 0; j < l; j++)
            if (j < w) {
                fputc(p[i * w + j], fp);
            } else {
                fputc(0, fp);
            }

    fclose(fp);

    return 0;
}
