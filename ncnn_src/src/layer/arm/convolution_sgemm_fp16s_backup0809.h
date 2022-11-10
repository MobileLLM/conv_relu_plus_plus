// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cpu.h"
#include "layer.h"
#include "mat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN


static __fp16 arr_product(const __fp16* tmpptr, const __fp16* kptr, int nn){
    __fp16 res = 0.f;
    for (int i=0; i<nn; i++){
        res += tmpptr[i] * kptr[i];
    }
    return res;

    int q = 0;
    float16x8_t _sum0 = vdupq_n_f16(0.f);
    for (; q + 7 < nn; q += 8)
    {
        float16x8_t _p0 = vld1q_f16(tmpptr);
        float16x8_t _k0 = vld1q_f16(kptr);
        _sum0 = vfmaq_f16(_sum0, _p0, _k0);

        tmpptr += 8;
        kptr += 8;
    }

    __fp16 sum0 = vaddvq_f32(vcvt_f32_f16(vadd_f16(vget_low_f16(_sum0), vget_high_f16(_sum0))));

    for (; q < nn; q++)
    {
        sum0 += tmpptr[0] * kptr[0];

        tmpptr++;
        kptr++;
    }

    return sum0;
}

static void sub_float_neon1(const __fp16* src1,const __fp16* src2, __fp16* dst, int nn)
{

    int q;
    for (q = 0; q + 7 < nn; q += 8)
    {
        float16x8_t in1, in2, out;
        in1 = vld1q_f16(src1);
        in2 = vld1q_f16(src2);

        out = vsubq_f16(in1, in2);
        vst1q_f16(dst, out);

        src1 += 8;
        src2 += 8;
        dst += 8;
    }

    for (; q < nn; q++)
    {
        dst[0] = src1[0] - src2[0];
        src1++;
        src2++;
        dst++;
    }
}

static __fp16 arr_delta_simd(const __fp16* src1, const __fp16* src2, int nn, const Option& opt){
    __fp16 res_vanilla = 0.f;
    for (int i=0; i<nn; i++){
        __fp16 diff = src1[i] - src2[i];
        res_vanilla += diff * diff;
    }
    return res_vanilla;

    Mat tmp;
    tmp.create(nn, 2u, 1, opt.workspace_allocator);
    __fp16* dst = (__fp16*)tmp.data;

    sub_float_neon1(src1, src2, dst, nn);
    __fp16 res = arr_product(dst, dst, nn);

    return res;
}


static void im2col_sgemm_fp16sa_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    size_t s = 2;

    const int outch = top_blob.c;
    const int nn = inch * maxk;

    Mat kernel = _kernel.reshape(maxk, inch, outch);

    const __fp16* bias = _bias;


    // permute
    Mat tmp;
    tmp.create(maxk, inch, size, 2u, 1, opt.workspace_allocator);


    //#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < size; i++)
    {
    __fp16* tmpptr = tmp.channel(i);
    for (int q = 0; q < inch; q++)
    {
        const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i;
        for (int k = 0; k < maxk; k++)
        {
            tmpptr[0] = img0[0];
            tmpptr += 1;
            img0 += size;
        }
    }
    }


    if (opt.use_reserved_0 && opt.use_reserved_3){
        Mat w_norm2(outch, s);
        __fp16* w_norm2_data = (__fp16*) w_norm2.data;
        for (int k=0; k<outch; k++){
            const __fp16* kptr = (const __fp16*) kernel.channel(k);
            w_norm2_data[k] = (__fp16) sqrt(arr_product(kptr, kptr, nn));
        }
        Mat delta_x_times_w_norm;
        delta_x_times_w_norm.create(size, 2u, 1, opt.workspace_allocator);
        Mat last_row_y;
        last_row_y.create(outch, 2u, 1, opt.workspace_allocator);

        /**
     * sampled Convolution
     */
        //    #pragma omp parallel for num_threads(opt.num_threads)
        __fp16* last_row_y_ptr = (__fp16*)last_row_y.data;

        for (int i=0; i < 1; i++){
            const __fp16* x_ij_ptr = tmp.channel(i);

            for (int p=0; p<outch; p++){
                Mat out0 = top_blob.channel(p);
                __fp16* outptr0 = out0;

                const __fp16 bias0 = bias ? bias[p] : 0.f;
                const __fp16* kptr = kernel.channel(p);
                __fp16 sum0 = arr_product(x_ij_ptr, kptr, nn) + bias0;
                outptr0[i] = sum0;
                last_row_y_ptr[0] = sum0;
                last_row_y_ptr++;
            }
        }

        auto gap = tmp.cstep * tmp.elemsize;
        __fp16* delta_ptr_ref = (__fp16*)delta_x_times_w_norm.data;
        const __fp16* x_ij_ptr_1 = tmp.channel(1);
        const __fp16* ref_ptr_1 =  tmp.channel(0);
        delta_ptr_ref++;
        for (int i=1; i<size; i++){

            delta_ptr_ref[0] = sqrt(arr_delta_simd(ref_ptr_1, x_ij_ptr_1, nn, opt));
            x_ij_ptr_1 = (__fp16*)((unsigned char*)x_ij_ptr_1 + gap);
            ref_ptr_1 =  (__fp16*)((unsigned char*)ref_ptr_1 + gap);
            delta_ptr_ref ++;
        }

        //
        float total = outch * size;
        float skipped = 0;
        //    #pragma omp parallel for num_threads(opt.num_threads)
        delta_ptr_ref = (__fp16*)delta_x_times_w_norm.data;
        delta_ptr_ref++;
        for (int i=1; i < size; i++){
            const __fp16* x_ij_ptr = tmp.channel(i);
            __fp16 delta_value = delta_ptr_ref[0];
            last_row_y_ptr = (__fp16*)last_row_y.data;

            for (int p=0; p<outch; p++){
                Mat out0 = top_blob.channel(p);
                __fp16* outptr0 = out0;

                __fp16 upper = last_row_y_ptr[0] + delta_value * w_norm2_data[p];
                if (upper <= 0.f){
                    outptr0[i] = upper;
                    last_row_y_ptr[0] = upper;
                    skipped += 1;
                }else{
                    const __fp16 bias0 = bias ? bias[p] : 0.f;
                    const __fp16* kptr = kernel.channel(p);
                    __fp16 sum0 = arr_product(x_ij_ptr, kptr, nn) + bias0;
                    outptr0[i] = sum0;
                    last_row_y_ptr[0] = sum0;
                }
                last_row_y_ptr++;
            }
            delta_ptr_ref++;
        }
        //
        fprintf(stderr, "%f\t", skipped/total);
//        printf( "our_conv %f\t", skipped/total);
//            fprintf(stderr, "%f\t%f\n", skipped/total, float(H)/outch);
    }else if(opt.use_reserved_0 && !opt.use_reserved_3){
        Mat last_tmp = tmp.clone();
        Mat last_y = top_blob.clone();
        for (int i=0; i < size; i++){
            const __fp16* x_ij_ptr = tmp.channel(i);

            for (int p=0; p<outch; p++){
                Mat out0 = top_blob.channel(p);
                __fp16* outptr0 = out0;

                const __fp16 bias0 = bias ? bias[p] : 0.f;
                const __fp16* kptr = kernel.channel(p);
                __fp16 sum0 = arr_product(x_ij_ptr, kptr, nn) + bias0;
                outptr0[i] = sum0;

            }
        }
    }
    else if (!opt.use_reserved_0){
        /**
     *  vanilla Convolution
     */
//        printf( "vanilla\n");
//            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i < size; i++){
                const __fp16* x_ij_ptr = tmp.channel(i);

                for (int p=0; p<outch; p++){
                    Mat out0 = top_blob.channel(p);
                    __fp16* outptr0 = out0;

                    const __fp16 bias0 = bias ? bias[p] : 0.f;
                    const __fp16* kptr = kernel.channel(p);
                    __fp16 sum0 = arr_product(x_ij_ptr, kptr, nn) + bias0;
                    outptr0[i] = sum0;

                }
            }
    }

}

static void convolution_im2col_sgemm_transform_kernel_fp16sa_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-8a-maxk-inch/8a-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(8 * maxk, inch, outch / 8 + outch % 8, (size_t)2u);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        __fp16* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 8; j++)
                {
                    const float* k00 = kernel.channel(q + j).row(p);

                    g00[0] = (__fp16)k00[k];

                    g00++;
                }
            }
        }
    }
    for (; q < outch; q++)
    {
        __fp16* g00 = kernel_tm.channel(q / 8 + q % 8);

        for (int p = 0; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k00 = kernel.channel(q).row(p);

                g00[0] = (__fp16)k00[k];

                g00++;
            }
        }
    }
}

static void convolution_im2col_sgemm_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 2u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            __fp16* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_fp16sa_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}


//  433 = 125.187500
//  615 = 108.750000
//  439 = 108.125000