// GreedyPermutations.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h> 
#include <stdlib.h> 
#include <emscripten/bind.h>
using namespace emscripten;




std::vector<int> getGreedyPerm(std::vector<std::vector<float> > points, int NPerm)
{
	std::vector<int> indices;
	//Creates vector of booleans whose values are true if the index has been added to indices or false if it hasn't been added yet
	//Defaulted to all false values
	std::vector<bool> visited(points.size(), false);

	indices.push_back(0); //Puts on the first index so our list isn't empty
	visited[0] = true; //Sets 0 to true since we have "visited" it

	float INFINITY = std::numeric_limits<float>::infinity();

	while (indices.size() < NPerm) {
		int K = indices.size();
		float minDistance = INFINITY;//Set to infinity so our first distance will be less than minDistance
		std::vector<float> minDistances;//List of mindistances from each column
		int minIndex;
		std::vector<int> minIndices;//Stores the indices of the point with the min distance associated with the equivalent place in the minDistances
		for (int i = 0; i < points.size(); i++) {

			for (int j = 0; j < K; j++) {
				//If points[j] has not been visited, calculate the distance
				//Otherwise we skip over j and move to the next point
				if (!visited[i]) {
					std::vector<float> point1 = points[i];
					std::vector<float> point2 = points[indices[j]];
					float distance = 0.0;
					for (int m = 0; m < point1.size(); m++) {
						distance += (point1[m] - point2[m]) * (point1[m] - point2[m]);
					}
					//Compares current distance to minDistance and replaces if smaller, if equal to 0, we compared the same points and we don't want to repeat
					if (distance < minDistance) {
						minDistance = distance;
						minIndex = i;
					}
				}
			}
			minIndices.push_back(minIndex);
			minDistances.push_back(minDistance);//adds the mindistance for this set to the list
		}
		//finds the max value of our minDistances
		float maxMinDistance = *std::max_element(minDistances.begin(), minDistances.end());
		//Finds the index of the max value
		std::vector<float>::iterator it = find(minDistances.begin(), minDistances.end(), maxMinDistance);
		int index = it - minDistances.begin();
		index = minIndices[index];
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
}