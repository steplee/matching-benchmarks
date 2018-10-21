
#include <utility>
#include "grid.h"

using namespace std;

int W = 1024, H = 1024;
float NOISE_PIX = .0001f;

static pair<vector<Vector2f>,vector<Vector2f>> genFakeData(int N=200) {
    vector<Vector2f> A{N,{0,0}}, B{N,{0,0}};

    for (int i=0; i<N; i++) {
        Vector2f a = Vector2f{ ((float)(rand() % 1000000) / 1000000) * W,
                               ((float)(rand() % 1000000) / 1000000) * H};
        Vector2f b = a + Vector2f{ ((float)(rand() % 1000000) / 1000000 - .5) * NOISE_PIX,
                                   ((float)(rand() % 1000000) / 1000000 - .5) * NOISE_PIX};

        A[i] = a; B[i] = b;
    }

    return {A,B};
}


int main() {

    int N = 512;
    int rows = 30, cols = 30, maxOccupancy = 1;

    vector<Grid*> grids = {
        new GridCpuNaive({"cpu"}, rows,cols,maxOccupancy),
        new GridCpuNaiveMultithreaded({"cpu_mt"}, rows,cols,maxOccupancy),
        new GridCudaNaive({"cuda"}, rows,cols,maxOccupancy)
    };

    int data_iters = 8;
    int iters = 3;

    for (int i=0; i<data_iters; i++) {
        int NN = N * 1<<i;
        std::cout << "\n ============================================== \n";
        std::cout << "        For data size " << NN << "\n";
        auto ab = genFakeData(NN);
        for (auto g : grids)
            for (int j=0; j<iters; j++)
                g->matchPoints(ab.first, ab.second);
    }

    return 0;
}
