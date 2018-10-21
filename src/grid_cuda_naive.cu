
#include "grid.h"

constexpr int BLOCK_SIZE = 512;

void GridCudaNaive::assignPointsToGrid(const std::vector<Vector2f>& pts) {

}


template<int Len>
__device__ float deviceDist(const float* ap, const float *bp) {
    float sum = 0.f;
    for (int i=0; i<Len; i++)
        sum += fabsf(ap[i] - bp[i]);
    return sum;
}

// Template specialization so that loops are unrolled. Maybe nvcc determines optimization itself?
template<> __device__ float deviceDist<1>(const float* ap, const float *bp) { return fabsf(ap[0] - bp[0]); }
template<> __device__ float deviceDist<2>(const float* ap, const float *bp) { return fabsf(ap[0] - bp[0]) + fabsf(ap[1] - bp[1]); }
template<> __device__ float deviceDist<3>(const float* ap, const float *bp) { return fabsf(ap[0] - bp[0]) + fabsf(ap[1] - bp[1]) + fabs(ap[2]-bp[2]); }


/* O(B) loop per A threads.
   Does NOT keep track of already-used ptsB (so may get cross-matches)
*/
__global__
void matchKernelSlow(
        const float* A, const float* B, const int Bn, const float distThresh,
        int* out) {

    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;

    for (int i=0; i<Bn; i++) {
        if (deviceDist<2>(A+tid*2, B+i*2) < distThresh)
            out[tid] = i;
    }
}

/*
   O(1) per A*B threads
   This is non-determinstic since I don't use atomics. That is okay if we don't have many
   false-matches.
   You could keep use harris score or just match-distance as a score and CAS any contentions.

   Also doesn't keep track of already used ptsB, hence why it is called 'Approixmate'
*/
__global__
void matchKernelFastApproximate(
        const float* A, const float* B, const int An, const int Bn, const float distThresh,
        int* out) {

    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    size_t idxA = tid / Bn;
    size_t idxB = tid % Bn;

    if (idxA < An and deviceDist<2>(A+idxA, B+idxB) < distThresh)
        out[idxA] = idxB;
}

/*
   Instead of having a single float thresh, we have a thresh per idxA, replaced when a new min dist is found.
   Unfortunately cuda only has atomic integral values, so you need to upscale then quantize floats (meaning we need
     the domain of possibilites beforehand, in this case *10000 will do fine).
   'AtomicOnDistance' because we use distance as the deciding factor, you could also use e.g. Harris score.
*/
constexpr float F2I_MULT = 10000;
template<int Stride>
__global__
void matchKernelFastAtomicOnDistance(
        const float* A, const float* B, const int An, const int Bn,
        int* dists,
        int* out) {

    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;

    // First line is faster but has contention: atomics seem to not work properly?
    //size_t idxA = tid / Bn, idxB = tid % Bn;
    size_t idxA = tid % Bn, idxB = tid / Bn;

    float dist_ = deviceDist<Stride>(A+idxA*Stride, B+idxB*Stride);
    int dist = (int) (dist_* F2I_MULT);
    int distThresh = atomicMin(dists+idxA, dist);


    if (dist <= distThresh) {
        //out[idxA] = idxB;
        //atomicMax(out+idxA, idxB);
        atomicExch(out+idxA, idxB);
    }
}

std::vector<int> GridCudaNaive::matchPoints(const std::vector<Vector2f>& ptsA, const std::vector<Vector2f>& ptsB) {
    boost::timer::cpu_timer t; t.start();

    // Copy to gpu. This could be avoided with managed cuda mem from the start.
    float *A, *B;
    cudaMalloc((void**)&A, sizeof(float)*ptsA.size()*2);
    cudaMalloc((void**)&B, sizeof(float)*ptsB.size()*2);
    cudaMemcpy(A, ptsA.data(), sizeof(float)*ptsA.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(B, ptsB.data(), sizeof(float)*ptsB.size()*2, cudaMemcpyHostToDevice);

    int *out;
    cudaMalloc((void**)&out, sizeof(int)*ptsA.size());
    cudaMemset(out, -1, ptsA.size()*sizeof(int));

    // ====================================================
    //     Slow, but correct.
    // ====================================================
    //matchKernelSlow<<<ptsA.size(), 1>>>( A, B, ptsB.size(), distThresh, out);


    int thr_per = min(BLOCK_SIZE, (int)ptsA.size());
    int blocks = ptsB.size() * (ptsA.size()/thr_per);

    // ====================================================
    //     Fastest, but incorrect.
    // ====================================================
    /*
    matchKernelFastApproximate<<<blocks, thr_per, 1>>>( A, B, ptsA.size(), ptsB.size(), distThresh, out);
    */

    // ====================================================
    //     Correct & Fast.
    // ====================================================
    if (scratchSize < ptsA.size()) {
        if (scratch) cudaFree(scratch);
        cudaMalloc((void**)&scratch, sizeof(float)*ptsA.size());
        scratchSize = ptsA.size();
    }

    int def_thresh = F2I_MULT * distThresh;
    cudaMemset(scratch, def_thresh, sizeof(int)*ptsA.size());
    matchKernelFastAtomicOnDistance<2><<<blocks, thr_per>>>( A, B, ptsA.size(), ptsB.size(), scratch, out);

    cudaDeviceSynchronize();

    std::vector<int> hostOut(ptsA.size(), -1);
    cudaMemcpy(hostOut.data(), out, sizeof(int)*ptsA.size(), cudaMemcpyDeviceToHost);

    cudaFree(A); cudaFree(B); cudaFree(out);

    /*
    for (int i=0; i<min(10,(int)hostOut.size()); i++) std::cout << hostOut[i] << ", "; std::cout << " ..., ";
    for (int i=hostOut.size()-10; i<hostOut.size(); i++) std::cout << hostOut[i] << ", "; std::cout << "\n\n";
    */

    print_timer(t);

    return hostOut;
}
GridCudaNaive::~GridCudaNaive() {
    if(scratch) {
        cudaFree(scratch);
        scratch = 0;
    }
}
