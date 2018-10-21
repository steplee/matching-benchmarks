
#include <vector>
#include <string>

#include <Eigen/Core>
#include <boost/timer/timer.hpp>

using Eigen::Vector2f;
using Eigen::Vector3f;

struct Grid
{
    inline Grid(std::string name, int rows, int cols, int maxOccupancy, float distThresh=.8) :
        name(name), rows(rows), cols(cols), maxOccupancy(maxOccupancy), distThresh(distThresh)
    {}

    int rows, cols, maxOccupancy;
    float distThresh;
    std::string name;

    virtual void assignPointsToGrid(const std::vector<Vector2f>& pts) =0;
    virtual std::vector<int> matchPoints(const std::vector<Vector2f>& ptsA, const std::vector<Vector2f>& ptsB) =0;

    inline void print_timer(boost::timer::cpu_timer& t, const char* task="one step") {
        t.stop();
        float millis = t.elapsed().wall * .000001;
        printf(" -- %s took %.2fms for %s.\n", name.c_str(), millis, task);
    }

    inline float dist(Vector2f a, Vector2f b) {
        return fabs(a(0) - b(0)) + fabs(a(1) - b(1));
    }
};

struct GridCpuNaive : public Grid {
    using Grid::Grid;
    virtual void assignPointsToGrid(const std::vector<Vector2f>& pts) override;
    virtual std::vector<int> matchPoints(const std::vector<Vector2f>& ptsA, const std::vector<Vector2f>& ptsB) override;
};

struct GridCpuNaiveMultithreaded : public Grid {
    using Grid::Grid;
    virtual void assignPointsToGrid(const std::vector<Vector2f>& pts) override;
    virtual std::vector<int> matchPoints(const std::vector<Vector2f>& ptsA, const std::vector<Vector2f>& ptsB) override;
};

struct GridCudaNaive : public Grid {
    //using Grid::Grid;
    virtual void assignPointsToGrid(const std::vector<Vector2f>& pts) override;
    virtual std::vector<int> matchPoints(const std::vector<Vector2f>& ptsA, const std::vector<Vector2f>& ptsB) override;

    int* scratch;
    int scratchSize;
    inline GridCudaNaive(std::string name, int rows, int cols, int maxOccupancy, float distThresh=.8) :
        Grid(name,rows,cols,maxOccupancy,distThresh),
        scratch(nullptr), scratchSize(0)
    {}
    ~GridCudaNaive();
};
