
#include "quad.h"
#include <cstdlib>
#include <iostream>
#include <map>

using namespace Eigen;
using namespace std;

using QuadT = Quad<5, int>;
using IndexT = map<int, Vector2f>;


void makeRandom(QuadT* q, IndexT& ind, int N=1000) {
    constexpr int M = 1000000;
    for (int i=0; i<N; i++) {
        Vector2f k{((float)(rand()%M))/M,((float)(rand()%M))/M};
        q->insert(k, i);
        ind[i] = k;
    }
}

int main() {

  //Quad<5, int> q(Vector2f{0,0}, Vector2f{1,1});
  QuadT* q;
  cudaMallocManaged(&q, sizeof(QuadT));
  new (q) QuadT(Vector2f{0,0}, Vector2f{1,1}); // placement new.

  //q->insert(Vector2f{.1,.1}, 1);
  IndexT index;
  makeRandom(q, index);

  cout << " =============== \n    Searching CPU. \n ===============\n";

  int buf[10];
  Vector2f k{.1,.1};
  int n = q->searchDist(k, .1f, buf, 10);

  for (int i=0; i<n;i++)
      std::cout << i << ": " << buf[i] << " (" << index[buf[i]].transpose()
                << ") dist " << (k-index[buf[i]]).norm() << ".\n";

  cout << " =============== \n    Searching CPU/GPU. \n ===============\n";

  q->searchDist_cpu_cuda(k, .1f, &k,1, buf);
  memset(buf, 0, sizeof(buf));

  for (int i=0; i<1;i++)
      std::cout << i << ": " << buf[i] << " (" << index[buf[i]].transpose()
                << ") dist " << (k-index[buf[i]]).norm() << ".\n";

  return 0;
}
