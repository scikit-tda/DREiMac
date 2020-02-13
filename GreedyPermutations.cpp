// GreedyPermutations.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <emscripten/bind.h>
using namespace emscripten;


std::vector<int> getGreedyPerm(std::vector<std::vector<float> > points, int NPerm)
{
	std::vector<int> indices;
	indices.push_back(0); //Puts on the first index so our list isn't empty
    float INFINITY = std::numeric_limits<float>::infinity();
	while (indices.size() < NPerm) {
		int K = indices.size();
		float minDistance = INFINITY;//Set to infinity so our first distance will be less than minDistance
		std::vector<float> minDistances;//List of mindistances from each column
		for (int i = 0; i < points.size();i++) {
			for (int j = 0; j < K;j++) {
				//Grabs current points
				std::vector<float> point1 = points[i];
				std::vector<float> point2 = points[indices[j]];
				float distance = 0.0;
				for (int m = 0; m < point1.size(); m++) {
					distance += (point1[m] - point2[m])*(point1[m] - point2[m]);
				}
				//Compares current distance to minDistance and replaces if smaller
				if (distance < minDistance) {
					minDistance = distance;
				}
			}
			minDistances.push_back(minDistance);//adds the mindistance for this set to the list
		}
		//finds the max value of our minDistances
		float maxMinDistance = *std::max_element(minDistances.begin(), minDistances.end());
		//Finds the index of the max value
		std::vector<float>::iterator it = find(minDistances.begin(), minDistances.end(), maxMinDistance);
		int index = std::distance(minDistances.begin(), it);
		//Add the index of the max of this list to the indices list
		indices.push_back(index);
	}

	return indices;
}

EMSCRIPTEN_BINDINGS(stl_wrappers) {
    emscripten::register_vector<float>("VectorFloat");
    emscripten::register_vector<int>("VectorInt");
    emscripten::register_vector<std::vector<float>>("VectorVectorFloat");
}

EMSCRIPTEN_BINDINGS(my_module) {
    function("getGreedyPerm", &getGreedyPerm);
}