#pragma once

#include <Eigen/StdVector>
#include <Eigen/Core>

template <int Cap, class V>
struct Leaf {
    int occupancy;
    Eigen::Vector2f keys[Cap];
    V vals[Cap];

   void init(); 
};

template<int Cap, class V>
class Quad {

    private:

        // Top left and bot right
        Eigen::Vector2f tl;
        Eigen::Vector2f br;
        Eigen::Vector2f mid;

        // Quadrants in order: 0 = tl, 1 = tr, 2 = bl, 3 = br
        Quad<Cap,V>* next[4];
        Leaf<Cap,V>* leaves[4];

    public:

        Quad(Eigen::Vector2f tl, Eigen::Vector2f br);
        ~Quad();

        bool insert(Eigen::Vector2f key, V val);

        // returns num written to buf.
        // Does not get in order, nor the closest (but any closest *enough*)
        int searchDist(Eigen::Vector2f key, float dist, V* buf, int bufN);

        // There are keysN keys and keysN bufs, each of length bufLen.
        void searchNN_cuda(Eigen::Vector2f* keys, int keysN,
                           int* bufs, int bufLen);


        // does a 1:1 matching.
        void searchDist_cpu_cuda(Eigen::Vector2f outerKey, float dist,Eigen::Vector2f* keys, int keysN,
                                 V* buf);

};

