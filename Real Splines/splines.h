typedef struct{
        float4 r1, r2, r3, r4;
} float4x4;
__DEVICE__ float4x4 make_float4x4(float4 row1, float4 row2, float4 row3, float4 row4){
    float4x4 Mat;
    Mat.r1 = row1, Mat.r2 = row2, Mat.r3 = row3, Mat.r4 = row4;
    return Mat;
}

__CONSTANT__ float4x4 catmull_rom =                         // C1 continuous, truly interpolating, local controll
    make_float4x4(make_float4( 0.f, 2.f, 0.f, 0.f) * 0.5f,
                  make_float4(-1.f, 0.f, 1.f, 0.f) * 0.5f,
                  make_float4( 2.f,-5.f, 4.f,-1.f) * 0.5f,
                  make_float4(-1.f, 3.f,-3.f, 1.f) * 0.5f);

__CONSTANT__ float4x4 B_spline =                            // C2 continuous (smooth as fuck), 
    make_float4x4(make_float4( 1.f, 4.f, 1.f, 0.f) / 6.f,
                  make_float4(-3.f, 0.f, 3.f, 0.f) / 6.f,
                  make_float4( 3.f,-6.f, 3.f, 0.f) / 6.f,
                  make_float4(-1.f, 3.f,-3.f, 1.f) / 6.f);

__CONSTANT__ float4x4 bezier =
    make_float4x4(make_float4( 1.f, 0.f, 0.f, 0.f),
                  make_float4(-3.f, 3.f, 0.f, 0.f),
                  make_float4( 3.f,-6.f, 3.f, 0.f),
                  make_float4(-1.f, 3.f,-3.f, 1.f));

__CONSTANT__ float4x4 hermite =
    make_float4x4(make_float4( 1.f, 0.f, 0.f, 0.f),
                  make_float4( 0.f, 1.f, 0.f, 0.f),
                  make_float4(-3.f,-2.f, 3.f,-1.f),
                  make_float4( 2.f, 1.f,-2.f, 1.f));

__DEVICE__ inline float dot4(float4 a, float4 b){          //normal dot produkt
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

__DEVICE__ inline float4 MatVekMult4(float4x4 mat, float4 vek){        // mult Matrix * Vektor
    float4 product = make_float4(dot4(mat.r1, vek),
                                 dot4(mat.r2, vek),
                                 dot4(mat.r3, vek),
                                 dot4(mat.r4, vek));
    return product;
}

__DEVICE__ float square(float x){
    return x * x;
}

__DEVICE__ float x_vs_spline(float4x4 SplineType, float x, float2 P1, float2 P2, float2 P3, float2 P4){
    
    float4 PointList_x = make_float4(P1.x, P2.x, P3.x, P4.x);
    float4 PointList_y = make_float4(P1.y, P2.y, P3.y, P4.y);

    float guess_t = x;
    float new_guess_t;
    float4 t_vektor;
    float4 t_abl;
    float tolerance = 0.00001f;
    float t_sq;

    for(int i = 0; i < 10; ++i){                //Newton-Raphson
        t_sq = square(guess_t);
        t_vektor = make_float4(1.f, guess_t, t_sq, t_sq * guess_t);
        t_abl = make_float4(0.f, 1.f, 2.f * guess_t, 3.f * t_sq);
        float f_readout = dot4(t_vektor, MatVekMult4(SplineType,PointList_x));
        float Abl_readout = dot4(t_abl, MatVekMult4(SplineType,PointList_x));

        new_guess_t = guess_t - ((f_readout - x) / Abl_readout);

        if(_fabs(new_guess_t - guess_t) < tolerance){
            break;
        } else{
            guess_t = new_guess_t;
        }
    }
    float y = dot4(t_vektor, MatVekMult4(SplineType,PointList_y));
    return y;
}

__DEVICE__ float assemble_6point_B_spline(float x, float2 P0, float2 P1, float2 P2, float2 P3, float2 P4, float2 P5){   //example function of how to piecewise define a spline with more control points (here a B-spline with 6 control points)
    float2 G0 = P0 - (P1 - P0);        //extrapolated points for start and finish
    float2 G1 = P5 + (P5 - P4);
    float2 G3 = P5 + 2.f * (P5 - P4); //for extension above 1.f

    if (x < P1.x){
        return x_vs_spline(B_spline, x, G0, P0, P1, P2);
    } else if (x < P2.x){
        return x_vs_spline(B_spline, x, P0, P1, P2, P3);
    } else if (x < P3.x){
        return x_vs_spline(B_spline, x, P1, P2, P3, P4);
    } else if (x < P4.x){
        return x_vs_spline(B_spline, x, P2, P3, P4, P5);
    } else if (x < P5.x){
        return x_vs_spline(B_spline, x, P3, P4, P5, G1);
    } else {
        return x_vs_spline(B_spline, x, P4, P5, G1, G3);
    }
}
