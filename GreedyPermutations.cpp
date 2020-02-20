// GreedyPermutations.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h> 
#include <stdlib.h> 
#include <emscripten/bind.h>
using namespace emscripten;


void clearVectorVector(std::vector<std::vector<float> >& M) {
    for (size_t i = 0; i < M.size(); i++) {
        M[i].clear();
    }
    M.clear();
}

/**
 * 
 * @param{points} Array of points in Euclidean space
 * @param{NPerm} Number of points to choose in permutation, should be <= #points
 * @param{distLandLand} An empty vector of vectors, passed by reference, that will
 *                      be used to store the inter-landmark distances
 * @param{distLandData} An empty vector of vectors, passed by reference, that will
 *                      be used to store the landmark to point cloud distances
 * */
std::vector<int> getGreedyPerm(std::vector<std::vector<float> > points, int NPerm, 
                               std::vector<std::vector<float> >& distLandLand,
                               std::vector<std::vector<float> >& distLandData )
{
    // Step 0: Clear distLandLand and distLandData if they had stuff in them
    // from before
	std::vector<int> indices;
	//Creates vector of booleans whose values are true if the index has been added to indices or false if it hasn't been added yet
	//Defaulted to all false values
	std::vector<bool> visited(points.size(), false);

	indices.push_back(0); //Puts on the first index so our list isn't empty
	visited[0] = true; //Sets 0 to true since we have "visited" it

	float INFINITY = std::numeric_limits<float>::infinity();

	while (indices.size() < NPerm) {
		int K = indices.size();
		float minDistance;
        std::vector<float> minDistances;//List of mindistances from each column
		for (int i = 0; i < points.size(); i++) {
            //Set to infinity so our first distance will be less than minDistance
            minDistance = INFINITY;
            std::vector<float> point1 = points[i];
            if (!visited[i]) {
			    for (int j = 0; j < K; j++) {
				    //If points[j] has not been visited, calculate the distance
				    //Otherwise we skip over j and move to the next point
					std::vector<float> point2 = points[indices[j]];
					float distance = 0.0;
					for (int m = 0; m < point1.size(); m++) {
						distance += (point1[m] - point2[m]) * (point1[m] - point2[m]);
					}
					//Compares current distance to minDistance and replaces if smaller, if equal to 0, we compared the same points and we don't want to repeat
					if (distance < minDistance) {
						minDistance = distance;
					}
			    }
                minDistances.push_back(minDistance);//adds the mindistance for this set to the list
            }
            else {
                minDistances.push_back(0);
            }			
		}
		//finds the max value of our minDistances
		float maxMinDistance = *std::max_element(minDistances.begin(), minDistances.end());
		//Finds the index of the max value
		std::vector<float>::iterator it = find(minDistances.begin(), minDistances.end(), maxMinDistance);
		int index = it - minDistances.begin();
		//Add the index of the max of this list to the indices list
		indices.push_back(index);
		//Set the index of our point to true since we have visited it
		visited[index] = true;
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
    function("clearVectorVector", &clearVectorVector);
}