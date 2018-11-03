#include "quad.h"
#include <iostream>
#include <stack>
#include <set>
#include <unistd.h>
#include <cstdio>

using namespace Eigen;
using namespace std;

template<int C, class V>
void Leaf<C,V>::init() {
    occupancy = 0;
}

template<int C, class V>
Quad<C,V>::Quad(Vector2f tl, Vector2f br)
    : tl(tl), br(br)
{
    mid = (tl + br) * .5f;

    //cudaMallocManaged(&next, sizeof(Quad<C,V>) * 4);
    //cudaMallocManaged(&leaves, sizeof(Leaf<C,V>) * 4);

    // Start with only 4 leaves.
    cudaMemset(next, 0, sizeof(void*) * 4);


    for (int i=0; i<4; i++) {
        cudaMallocManaged(&leaves[i], sizeof(Leaf<C,V>));
        leaves[i]->init();
    }

}

template<int C, class V>
Quad<C,V>::~Quad() {
    for (int i=0; i<4; i++) if (next[i]) {
        cudaFree(next[i]);
        next[i] = 0;
    }
    for (int i=0; i<4; i++) if (leaves[i]) {
        cudaFree(leaves[i]);
        leaves[i] = 0;
    }
}

template<int C, class V>
bool Quad<C,V>::insert(Eigen::Vector2f key, V val) {
    uint8_t ord = 0;
    if (key(0) >= mid(0)) ord |= 1;
    if (key(1) >= mid(1)) ord |= 2;

    cout << " key " << key.transpose() << " mid " << mid.transpose() << "\n";
    cout << " ord " << (int) ord << "\n";

    // If true, we already split the leaf, so insert to next quad.
    if (leaves[ord] == nullptr) {
        assert(next[ord]);
        return next[ord]->insert(key, val);
    }

    int &j = leaves[ord]->occupancy;

    if (j == C) {
        // TODO split & create nodes.
        cudaMallocManaged(&next[ord], sizeof(Quad<C,V>));
        Vector2f ntl, nbr;
        if (ord == 0) ntl = tl, nbr = mid;
        else if (ord == 1) ntl = {mid(0),tl(1)}, nbr = {br(0),mid(1)};
        else if (ord == 2) ntl = {tl(0),mid(1)}, nbr = {mid(0),br(1)};
        else if (ord == 3) ntl = mid, nbr = br;
        new (next[ord]) Quad<C,V>(ntl,nbr); // placement new.

        // node created, now redistribute what was in leaf, then delete it.
        for (int i=0; i<j; i++) {
            next[ord]->insert(leaves[ord]->keys[i], leaves[ord]->vals[i]);
        }
        cudaFree(leaves[ord]);
        leaves[ord] = 0;

        return true;
    }

    // Else we just add to leaf
    leaves[ord]->keys[j] = key;
    leaves[ord]->vals[j] = val;
    j++;
    return true;

    /*
    if (next[ord] == nullptr) {
        //next[ord] = new Quad ...
        cudaMallocManaged(&next[ord], sizeof(Quad<C,V>));
    }
    */
}

// search

template<int C, class V>
int Quad<C,V>::searchDist(Eigen::Vector2f key, float dist, V* buf, int bufN) {
    // When searching, you have to inspect all cells that are at most [nns[-1]] away.
    // i.e. you have to look at every point in every lowest-cell that is closer than the farthest neighbor

    std::vector<Leaf<C,V>*> closeLeaves;

    auto close = [&](Quad<C,V>* q) {
        if (key(0) > q->tl(0) and key(0) < q->br(0) and key(1) > q->tl(1) and key(1) < q->br(1)) return true;
        float k2m = (q->mid - key).norm() - (q->tl(0)-q->mid(0)) / 1.414;
        return dist - k2m >= 0;
    };


    // 1. dfs.
    stack<Quad<C,V>*> st;
    st.push(this);
    {
        while (not st.empty()) {
            auto node = st.top(); st.pop();
            for (auto q : node->next) if (q and close(q)) st.push(q);
            for (auto l : node->leaves) if (l and close(node)) closeLeaves.push_back(l);
        }
    }

    // 2. verify
    int bufi=0;
    for (auto l : closeLeaves) {
        for (int i=0; i<l->occupancy; i++)
            if ((l->keys[i] - key).norm() < dist) {
                buf[bufi++] = l->vals[i];
                if (bufi >= bufN) return bufi;
            }
    }

    return bufi;

}

template<int C, class V>
void Quad<C,V>::searchNN_cuda(Eigen::Vector2f* keys, int keysN,
                              int* bufs, int bufLen) {

    // Obvious thing is to have a thread per qkey and traverse tree in threads. Worried about memory access there though.

    // What about this strategy?
    // Give one key from pose (with distance for entire frame), then accumulate all closest leaves (cpu!), then (gpu) match closest/best-score to qkey-set?
    // Ofc you could do qkey-set to result as well.
    // This accumulation could collect leaf pointers to an array, then have the kernel iterate over each,
    // OR you could launch a kernel for each in parallel and use atomics to get best.

    // <<<>>>
}

template<int C, class V>
__global__ void searchFineCuda(Leaf<C,V>** leaves, int L, Vector2f* keys,
        V* out) {

    size_t tid = threadIdx.x  + blockIdx.x * blockDim.x;
    printf("tid %lu\n",tid);

    float bestD = 99999;
    for (int i=0; i<L; i++) {
        printf("l %d (sz %d)\n",i, leaves[i]->occupancy);
        for (int j=0; j<leaves[i]->occupancy; j++) {
            float d = (leaves[i]->keys[j] - keys[tid]).norm();
            printf("%d %d : %.2f\n",i,j,d);
            if (d < bestD) {
                bestD = d;
                out[tid] = leaves[i]->vals[j];
            }
        }
    }

}

template<int C, class V>
void Quad<C,V>::searchDist_cpu_cuda(Vector2f outerKey, float dist,Eigen::Vector2f* keys, int keysN,
                                 V* buf) {


    std::vector<Leaf<C,V>*> closeLeaves;

    auto close = [&](Quad<C,V>* q) {
        if (outerKey(0) > q->tl(0) and outerKey(0) < q->br(0) and outerKey(1) > q->tl(1) and outerKey(1) < q->br(1)) return true;
        float k2m = (q->mid - outerKey).norm() - (q->tl(0)-q->mid(0)) / 1.414;
        return dist - k2m >= 0;
    };


    // 1. dfs.
    stack<Quad<C,V>*> st;
    st.push(this);
    {
        while (not st.empty()) {
            auto node = st.top(); st.pop();
            for (auto q : node->next) if (q and close(q)) st.push(q);
            for (auto l : node->leaves) if (l and close(node)) closeLeaves.push_back(l);
        }
    }

    cout << "   c/gpu has " << closeLeaves.size() << " closeLeaves.\n";

    // TODO doesnt work

    Leaf<C,V>** the_leaves;
    cudaMallocManaged(&the_leaves, sizeof(void*) * closeLeaves.size());
    cudaMemcpy(the_leaves, (char*)closeLeaves.data(), sizeof(void*)*closeLeaves.size(), cudaMemcpyHostToHost);
    for (int i=0; i<closeLeaves.size(); i++) std::cout << closeLeaves[i] << " " << the_leaves[i] << "\n";
    searchFineCuda<<<1,1>>>(the_leaves, closeLeaves.size(),
                                keys, buf);

    cudaFree(the_leaves);

    usleep(1000000);
}

// Instantiations.
template class Quad<5, int>;
//template __global__ void searchFineCuda<5, int>(Leaf<5,int>** leaves, int L, Vector2f* keys, int* out);
