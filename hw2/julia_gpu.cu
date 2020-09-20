/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <iostream>
#include <chrono>

#define DIM 1000

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void blur(unsigned char* in_bm, unsigned char* out_bm) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int maxGridDim = gridDim.x;
    int xmin = -1;
    int xmax = 1;
    int ymin = -1;
    int ymax = 1;
    
    if (x == 0) {
        xmin = 0; 
    } 
    if (x == maxGridDim) { 
        xmax = 0;
    }
    if (y == 0) {
        ymin = 0;
    }
    if (y == maxGridDim) {
        ymax = 0; 
    }

    float val = 0;

    for (int i = ymin; i <= ymax; i++) {
        for (int j = xmin; j <= xmax; j++) {
            int index = x + j + (y + i) * DIM;
	    val += in_bm[index * 4]/9.0;
        }
    }

    out_bm[offset*4] = int(val);
    out_bm[offset*4 + 1] = 0;
    out_bm[offset*4 + 2] = 0;
    out_bm[offset*4 + 3] = 255;
    
}

__global__ void sharpen(unsigned char *in_bm, unsigned char* out_bm) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = (x + y * gridDim.x);
    int xmin = -1;
    int xmax = 1;
    int ymin = -1;
    int ymax = 1;
    int maxGridDim = gridDim.x;
    
    if (x == 0) {
        xmin = 0; 
    } 
    if (x == maxGridDim) { 
        xmax = 0;
    }
    if (y == 0) {
        ymin = 0;
    }
    if (y == maxGridDim) {
        ymax = 0; 
    }

    float val = 0; 

    for (int row = ymin; row <= ymax; row++) {
        for (int col = xmin; col <= xmax; col++) {      
            int index = x + col + (y + row) * maxGridDim;
            double multiplier = 0.0f;
	    if (col == 0 && (row == -1 || row == 1)) {
		multiplier = -0.5f;
	    }
            if (row == 0 && (col == -1 || col == 1)) {
		multiplier = -0.5f;
	    }
            if (row == 0 && col == 0) {
		multiplier = 3.0f;
	    }
            val += in_bm[index * 4] * multiplier;
        }
    }

    out_bm[offset*4] = int(val);
    out_bm[offset*4 + 1] = 0;
    out_bm[offset*4 + 2] = 0;
    out_bm[offset*4 + 3] = 255;
}

__global__ void kernel( unsigned char *ptr ) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue = julia( x, y );
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    dim3    grid(DIM,DIM);

    DataBlock   orig_data;
    CPUBitmap orig_bitmap( DIM, DIM, &orig_data );
    unsigned char    *orig_dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&orig_dev_bitmap, orig_bitmap.image_size() ) );
    orig_data.dev_bitmap = orig_dev_bitmap;
    kernel<<<grid, 1>>>( orig_dev_bitmap );

    // Setting up Blurred image bitmap
    DataBlock   blur_data;
    CPUBitmap blur_bitmap( DIM, DIM, &blur_data );
    unsigned char    *blur_dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&blur_dev_bitmap, blur_bitmap.image_size() ) );
    blur_data.dev_bitmap = blur_dev_bitmap;
    blur<<<grid, 1>>>( orig_dev_bitmap, blur_dev_bitmap );

    // Setting up Sharpened image bitmap
    DataBlock   sharp_data;
    CPUBitmap sharp_bitmap( DIM, DIM, &sharp_data );
    unsigned char    *sharp_dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&sharp_dev_bitmap, sharp_bitmap.image_size() ) );
    sharp_data.dev_bitmap = sharp_dev_bitmap;
    sharpen<<<grid, 1>>>( orig_dev_bitmap, sharp_dev_bitmap );

    // Handle memory
    HANDLE_ERROR( cudaMemcpy( orig_bitmap.get_ptr(), orig_dev_bitmap, orig_bitmap.image_size(), cudaMemcpyDeviceToHost ) );    
    HANDLE_ERROR( cudaFree( orig_dev_bitmap ) );
    HANDLE_ERROR( cudaMemcpy( blur_bitmap.get_ptr(), blur_dev_bitmap, blur_bitmap.image_size(), cudaMemcpyDeviceToHost ) );    
    HANDLE_ERROR( cudaFree( blur_dev_bitmap ) );
    HANDLE_ERROR( cudaMemcpy( sharp_bitmap.get_ptr(), sharp_dev_bitmap, sharp_bitmap.image_size(), cudaMemcpyDeviceToHost ) );    
    HANDLE_ERROR( cudaFree( sharp_dev_bitmap ) );

    //orig_bitmap.display_and_exit();         
    //blur_bitmap.display_and_exit();         
    sharp_bitmap.display_and_exit();         
}

