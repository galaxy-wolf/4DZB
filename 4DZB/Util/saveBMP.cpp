#include "SaveBMP.h"
void SaveBMP1(const char* fileName, BYTE * buf, UINT width, UINT height)
{
	short res1 = 0;
	short res2 = 0;
	long pixoff = 54;
	long compression = 0;
	long cmpsize = 0;
	long colors = 0;
	long impcol = 0;
	char m1 = 'B';
	char m2 = 'M';

	DWORD widthDW = WIDTHBYTES(width * 24);

	long bmfsize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) +
		widthDW * height;
	long byteswritten = 0;

	BITMAPINFOHEADER header;
	header.biSize = 40;
	header.biWidth = width;
	header.biHeight = height;
	header.biPlanes = 1;
	header.biBitCount = 24;
	header.biCompression = BI_RGB;
	header.biSizeImage = 0;
	header.biXPelsPerMeter = 0;
	header.biYPelsPerMeter = 0;
	header.biClrUsed = 0;
	header.biClrImportant = 0;

	FILE *fp;
	fp = fopen(fileName, "wb");
	if (fp == NULL)
	{
		//MessageBox("error","Can't open file for writing");
		return;
	}

	BYTE *topdown_pixel = (BYTE *)malloc(width*height * 3 * sizeof(BYTE));
	for (int j = 0; j < height; j++)
		for (int k = 0; k < width; k++)
		{
		memcpy(&topdown_pixel[(j*width + k) * 3], &buf[(j*width + k) * 3 + 2], sizeof(BYTE));
		memcpy(&topdown_pixel[(j*width + k) * 3 + 2], &buf[(j*width + k) * 3], sizeof(BYTE));
		memcpy(&topdown_pixel[(j*width + k) * 3 + 1], &buf[(j*width + k) * 3 + 1], sizeof(BYTE));
		}
	buf = topdown_pixel;

	//Ìî³äBITMAPFILEHEADER
	fwrite((BYTE  *)&(m1), 1, 1, fp); byteswritten += 1;
	fwrite((BYTE  *)&(m2), 1, 1, fp); byteswritten += 1;
	fwrite((long  *)&(bmfsize), 4, 1, fp);	byteswritten += 4;
	fwrite((int  *)&(res1), 2, 1, fp); byteswritten += 2;
	fwrite((int  *)&(res2), 2, 1, fp); byteswritten += 2;
	fwrite((long  *)&(pixoff), 4, 1, fp); byteswritten += 4;

	//Ìî³äBITMAPINFOHEADER
	fwrite((BITMAPINFOHEADER *)&header, sizeof(BITMAPINFOHEADER), 1, fp);
	byteswritten += sizeof(BITMAPINFOHEADER);


	//Ìî³äÎ»Í¼Êý¾Ý
	long row = 0;
	long rowidx;
	long row_size;
	row_size = header.biWidth * 3;
	long rc;
	for (row = 0; row < header.biHeight; row++) {
		rowidx = (long unsigned)row*row_size;

		// Ð´Ò»ÐÐ
		rc = fwrite((void  *)(buf + rowidx), row_size, 1, fp);
		if (rc != 1)
		{
			break;
		}
		byteswritten += row_size;

		for (DWORD count = row_size; count < widthDW; count++) {
			char dummy = 0;
			fwrite(&dummy, 1, 1, fp);
			byteswritten++;
		}

	}

	fclose(fp);
}

void saveScreenToBMP(const std::string & filename, int screenW, int screenH, bool halfCompress)
{
	if (!halfCompress)
	{
		BYTE* pixels = new BYTE[3 * screenW * screenH];

		glReadPixels(0, 0, screenW, screenH, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		SaveBMP1((filename + std::string(".bmp")).c_str(), pixels, screenW, screenH);
		delete[] pixels;
		return;
	}

	
	
	int outWidth = screenW >> 1;
	int outHeight = screenH >> 1;

	float* srcPixels = new float[3 * screenW * screenH];
	BYTE* pixels = new BYTE[3 * outWidth * outHeight];

	glReadPixels(0, 0, screenW, screenH, GL_RGB, GL_FLOAT, srcPixels);

	for (int i = 0; i < outHeight; i++)
	{
		for (int j = 0; j < outWidth; j++)
		{
			UCHAR res[3];
			for (int k = 0; k < 3; k++)
			{
				float a = 0;
				int ii = i * 2;
				int jj = j * 2;

				a += srcPixels[(ii*screenW + jj) * 3 + k];
				a += srcPixels[(ii *screenW + jj + 1) * 3 + k];
				a += srcPixels[((ii + 1) *screenW + jj) * 3 + k];
				a += srcPixels[((ii + 1) *screenW + jj + 1) * 3 + k];

				a /= 4;
				pixels[(i*outWidth + j) * 3 + k] = a * 255;
			}
		}

			////for (int k = 0; k < 3; k++)
			//	pixels[(i*outWidth + j) * 3 + 0] = res[2];
			//	pixels[(i*outWidth + j) * 3 + 1] = res[1];
			//	pixels[(i*outWidth + j) * 3 + 2] = res[0];
	}
	SaveBMP1((filename + std::string(".bmp")).c_str(), pixels, outWidth, outHeight);
	delete[] srcPixels;
	delete[] pixels;	
}

