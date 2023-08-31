#include <iostream>
#include <vector>
#include <map>
#include <cmath>

using namespace std;

struct DataPoint {
    double feature1;
    double feature2;
    int label;
};

// Calculate Gini impurity
double calculateGiniImpurity(const vector<DataPoint>& data) {
    int totalSamples = data.size();
    map<int, int> classCounts;

    for (const DataPoint& dp : data) {
        classCounts[dp.label]++;
    }

    double giniImpurity = 1.0;
    for (const auto& pair : classCounts) {
        double p = static_cast<double>(pair.second) / totalSamples;
        giniImpurity -= p * p;
    }

    return giniImpurity;
}

// Split data based on a feature and threshold
pair<vector<DataPoint>, vector<DataPoint>> splitData(const vector<DataPoint>& data, int featureIndex, double threshold) {
    vector<DataPoint> left, right;
    
    for (const DataPoint& dp : data) {
        if (dp.feature1 <= threshold) {
            left.push_back(dp);
        } else {
            right.push_back(dp);
        }
    }
    
    return make_pair(left, right);
}

// Recursive function to build the decision tree
void buildDecisionTree(const vector<DataPoint>& data, int depth) {
    const int maxDepth = 2;  // Set maximum depth for the tree

    if (depth >= maxDepth || data.empty()) {
        // Create leaf node and make classification
        // For simplicity, let's just print the predicted class
        int classCount[2] = {0, 0};
        for (const DataPoint& dp : data) {
            classCount[dp.label]++;
        }

        if (classCount[0] > classCount[1]) {
            cout << "Predicted Class: 0" << endl;
        } else {
            cout << "Predicted Class: 1" << endl;
        }
        return;
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGiniImpurity = 1.0;

    for (int featureIndex = 0; featureIndex < 2; ++featureIndex) {
        for (const DataPoint& dp : data) {
            double threshold = dp.feature1;
            auto [left, right] = splitData(data, featureIndex, threshold);

            double weightedGiniImpurity = (left.size() * calculateGiniImpurity(left) + 
                                           right.size() * calculateGiniImpurity(right)) / data.size();

            if (weightedGiniImpurity < bestGiniImpurity) {
                bestGiniImpurity = weightedGiniImpurity;
                bestFeature = featureIndex;
                bestThreshold = threshold;
            }
        }
    }

    auto [left, right] = splitData(data, bestFeature, bestThreshold);

    // Print decision node details
    cout << "Decision Node: Feature " << bestFeature + 1 << ", Threshold: " << bestThreshold << endl;

    // Recurse for left and right branches
    cout << "Left Branch:" << endl;
    buildDecisionTree(left, depth + 1);

    cout << "Right Branch:" << endl;
    buildDecisionTree(right, depth + 1);
}

int main() {
    // Sample dataset
    vector<DataPoint> data = {
        {2.0, 3.0, 0},
        {4.0, 5.0, 1},
        {1.5, 2.0, 0},
        {3.5, 4.5, 1},
        {2.5, 3.5, 0},
        {5.0, 6.0, 1},
    };

    // Build the decision tree
    buildDecisionTree(data, 0);

    return 0;
}
