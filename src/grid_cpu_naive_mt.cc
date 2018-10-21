#include "grid.h"
#include <future>
#include <thread>

void GridCpuNaiveMultithreaded::assignPointsToGrid(const std::vector<Vector2f>& pts) {

}

std::vector<int> GridCpuNaiveMultithreaded::matchPoints(const std::vector<Vector2f>& ptsA, const std::vector<Vector2f>& ptsB) {
    boost::timer::cpu_timer t; t.start();

    std::vector<std::atomic<bool>> bUsed(ptsB.size());
    for (int t=0; t<ptsB.size(); t++) bUsed[t] = false;

    int THREADS = 8;

    std::vector<int> divs;
    std::vector<std::future<std::vector<int>>> futs;

    for (int t=0; t<THREADS; t++) {
      futs.push_back(std::async(std::launch::async, [&ptsA, &ptsB, &bUsed, t, THREADS, this]() {
              int start = (t)*ptsA.size() / THREADS;
              int end = t == THREADS-1 ? ptsA.size() : (t+1)*ptsA.size() / THREADS;
              //printf(" -- mt %d->%d, thread_id %lu.\n", start, end, std::hash<std::thread::id>{}(std::this_thread::get_id()));
              std::vector<int> out(end-start, -1);
              for(int i=0; i<end-start; i++) {
                  const Vector2f& a = ptsA[i+start];

                  for (int j=0; j<ptsB.size(); j++) {
                      if (not bUsed[j].load() and dist(a, ptsB[j]) < distThresh) {
                          bUsed[j].store(true);
                          out[i] = j;
                          break;
                      }
                    }
              }
              return out;
          }));
    }

    std::vector<int> outs(ptsA.size(), -1);
    int ii = 0;
    for (auto &fut : futs) {
        auto vec = fut.get();
        //std::cout << "v " << vec.size() << "\n";
        for (auto z : vec) outs[ii++] = z;
    }


    print_timer(t);

    return outs;
}
