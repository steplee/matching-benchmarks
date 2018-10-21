#include "grid.h"

void GridCpuNaive::assignPointsToGrid(const std::vector<Vector2f>& pts) {

}

std::vector<int> GridCpuNaive::matchPoints(const std::vector<Vector2f>& ptsA, const std::vector<Vector2f>& ptsB) {
    boost::timer::cpu_timer t; t.start();

    std::vector<bool> bUsed(ptsB.size(), false);
    std::vector<int> out(ptsA.size(), -1);

    for (int i=0; i<ptsA.size(); i++) {
        const Vector2f& a = ptsA[i];

        for (int j=0; j<ptsB.size(); j++) {
            if (not bUsed[j] and dist(a, ptsB[j]) < distThresh) {
                bUsed[j] = true;
                out[i] = j;
                break;
            }
        }
    }

    print_timer(t);

    return out;
}
