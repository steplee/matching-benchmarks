#include <Eigen/Core>
#include <vector>
#include <cstdlib>
#include <iostream>

using namespace Eigen;
using namespace std;

int main() {

  vector<Vector2f> pts;

  int N = 4;

  float *_data;
  cudaMallocManaged( (void**)&_data, sizeof(float)*N*2 );
  for (int i=0; i<N; i++)
    memcpy(_data+i*2, Vector2f{1,0}.data(), sizeof(float)*2);

  for (int i=0; i<N; i++)
    std::cout << *(Vector2f*)(_data+i*2) << endl;


  vector<Vector2f> kpts(N);
  cudaMallocManaged(&kpts.data(), sizeof(float)*N*2);



  return 0;
}
