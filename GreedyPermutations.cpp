// GreedyPermutations.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;


int main()
{
    std::cout << "Hello World!\n";
}

vector<int> getGreedyPerm(vector<vector<double> > points, int NPerm)
{
	vector<int> indices;
	indices.push_back(0); //Puts on the first index so our list isn't empty
	while (indices.size() < NPerm) {
		int K = indices.size();
		double minDistance = INFINITY;//Set to infinity so our first distance will be less than minDistance
		vector<double> minDistances;//List of mindistances from each column
		for (int i = 0; i < points.size();i++) {
			for (int j = 0; j < K;j++) {
				//Grabs current points
				vector<double> point1 = points[i];
				vector<double> point2 = points[indices[j]];
				double distance = 0.0;
				for (int m = 0; m < point1.size(); m++) {
					distance += pow(point1[m] - point2[m], 2);
				}
				distance = sqrt(distance);//distance caluclated for current points
				//Compares current distance to minDistance and replaces if smaller
				if (distance < minDistance) {
					minDistance = distance;
				}
			}
			minDistances.push_back(minDistance);//adds the mindistance for this set to the list
		}
		//finds the max value of our minDistances
		double maxMinDistance = max_element(minDistances.begin, minDistances.end);
		//Finds the index of the max value
		vector<double>::iterator it = find(minDistances.begin, minDistances.end, maxMinDistance);
		int index = std::distance(minDistances.begin(), it);
		//Add the index of the max of this list to the indices list
		indices.push_back(index);
	}

	return indices;
}