#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <memory>
#include <random>
#include <chrono>

using namespace std;

struct Node {
    string featureName; //string to represent the name of feature used for splitting
    int featureIndex;  //column index of that feature in the data
    string label;      //string label if the node is a leaf
    bool isLeaf;       //true if this node is a leaf node (if it is a leaf then that node stores the final answer)
    map<string, Node*> children; //map used to connect feature values to the child nodes so that the tree knows which branch to follow
    
    Node() : featureIndex(-1), isLeaf(false) {} //constructor that runs when a new node is created
                                                //It sets the featureIndex = -1 and sets isLeaf = false (assuming it's not a leaf at first)
    
    ~Node() { //destructor that runs when the node is destroyed
        for(auto& pair : children) { //loops through every (key,value) in the children map
            delete pair.second;  //deletes each child node
        }
    }
};

void splitData(const vector<vector<string>>& data,
               vector<vector<string>>& trainData,
               vector<vector<string>>& testData,
               double trainRatio = 0.7) {

    vector<vector<string>> temp = data; //make a copy of the whole dataset
    random_device rd; //random number generator that is used to randomly shuffle the copied data
    mt19937 g(rd()); 
    shuffle(temp.begin(), temp.end(), g);

    int trainSize = (int)(temp.size() * trainRatio); //calculate how many samples go into training

    for (int i = 0; i < (int)temp.size(); i++) { //loop through every row in the shuffled dataset
        if (i < trainSize)  //check if the row's index is less than the training split
             trainData.push_back(temp[i]); //if it is, enter the training set
        else  //if not
            testData.push_back(temp[i]); //the training rows go into the test set
    }
}

vector<vector<string>> loadDataFile(const string& filename, char delimiter=',') {
    ifstream file(filename);
    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    vector<vector<string>> data;
    string line;
    
    while (getline(file, line)) { //check if the file failed to open
        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == string::npos) { //skip empty or whitespace only lines
            continue;
        }
        
        vector<string> row; //vector to store column values for this line
        stringstream ss(line); //convert the line into a stringstream so we can read each cell 
        string cell;
        
        while (getline(ss, cell, delimiter)) {
            // Trim whitespace but KEEP empty cells
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            
            // Add the cell even if it's empty
            row.push_back(cell);
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    return data;
}


//Gini Index
double giniImpurity(const vector<vector<string>>& data, int labelIndex) { //function to measure how mixed the classes are in a dataset
    if(data.empty()) //check to see if the data is empty
        return 0.0;
    
    map<string,int> counts; //map to store the class label and the number of times this label appears in the dataset
    for(const auto& row : data) { //count how many samples belong to each class
        counts[row[labelIndex]]++;  //increment the count for that label
    }
    
    double gini = 1.0; //initalizing the gini impurity to 1
    int total = data.size(); //stores the total number of samples in the dataset
    for(const auto& kv : counts) { //loop over each class in counts 
                                   // kv.first = class label 
                                   // kv.second = count of samples in that class
        double p = (double)kv.second / total;  // p = count/total gives us the total fractions of samples in that class
        gini -= p * p;  // subtract the squared probability from 1 (per the gini formula)
    }
    return gini; //return the gini impurity for this dataset/node
}

double giniSplit(const vector<vector<string>>& data, int featureIndex, int labelIndex) { //function how pure the data is after splitting by a feature
    map<string, vector<vector<string>>> subsets; //create a map that stores the feature value and the rows that have that value
    for(const auto& row : data) { //loop through every row in the dataset
        subsets[row[featureIndex]].push_back(row); //for the current row, look at the value of the chosen feature use it as a key in the map
    }
    
    double weightedGini = 0.0; //variable to store the final impurity value
    int total = data.size(); //int variable to store the total number of rows before splitting
    for(const auto& kv : subsets) { //loop through each group in the map
        double weight = (double)kv.second.size() / total; //calculate the weight of the group
        weightedGini += weight * giniImpurity(kv.second, labelIndex); //multiply it by the group's weight and then add the result to the total
    }
    return weightedGini; //return the final impurity score (the lower the value = better feature to split on)
}

//Information Gain 
double entropy(const vector<vector<string>>& data, int labelIndex) {
    if(data.empty()) //if the dataset is empty, return 0.0
        return 0.0;
    
    map<string,int> counts; //map to store the label and how many time it appears
    for(const auto& row : data) { //loop throuh every row and look at the label column
        counts[row[labelIndex]]++; //increase the count for that label
    }
    
    double entropyValue = 0.0; 
    int total = data.size();//store the total number fo rows in total
    for(const auto& kv : counts) { //loop over the entries in the counts map
        double p = (double)kv.second / total; //calculate the probability of the current label (count/total)
        if(p > 0) { //if p is greater than 0, compute the formula
                    //if not, don't
            entropyValue -= p * log2(p);
        }
    }
    return entropyValue; //return the final entropy value computed
}

double informationGain(const vector<vector<string>>& data, int featureIndex, int labelIndex) {  //function that picks the feature with the highest information gain to split the data
    double totalEntropy = entropy(data, labelIndex); //call entropy function to find out how mixed the class labels are before any splitting
    
    map<string, vector<vector<string>>> subsets; //map that stores the feature value and the number of rows with that value
    for(const auto& row : data) { //loop through every row and look at the value in the column we are splitting on
                                  //use that avlue as a key in the map
        subsets[row[featureIndex]].push_back(row); //add the full row into the group
    }
    
    double subsetEntropy = 0.0;
    int total = data.size(); //int variable to store the total number of rows in the dataset
    for(const auto& kv : subsets) { //loop through each group 
        double weight = (double)kv.second.size() / total; // calculate the weight of this group
        subsetEntropy += weight * entropy(kv.second, labelIndex); //multiply it by the group's weight and add it to the running total
    }
    
    return totalEntropy - subsetEntropy; //subtract the entropy befor ethe slide and the entropy after the split 
                                         //this gives us the information gain
}


//function that will be used in the gainRatio metric
double splitInfo(const vector<vector<string>>& data, int featureIndex) {
    return entropy(data, featureIndex); //call entropy function of the feature column
}

double gainRatio(const vector<vector<string>>& data, int featureIndex, int labelIndex) {
    double infoGain = informationGain(data, featureIndex, labelIndex); //call informationGain function and store how much the feature reudesc the label uncertainity
    double splitInformation = splitInfo(data, featureIndex); //call splitInfo to measure how spread out the feature values are

    if (splitInformation == 0.0 || splitInformation < 1e-10) //safety check to avoid dividing by zero
        return 0.0;                                          //if splitInformatoin is zero or extremely small -> return 0

    return infoGain / splitInformation; //Gain ratio = information gain / split info
}

string majorityLabel(const vector<vector<string>>& data, int labelIndex) { //function to find and return the label that appears the most in the dataset
    map<string,int> counts; //map to store the label and the number of times it appears
    for(const auto& row : data) { //loop throuh every row in the dataset
        counts[row[labelIndex]]++; //Increase teh value in the label column by 1
    }
    
    string majLabel;
    int maxCount = 0;
    for(const auto& kv : counts) { //loop through every label in the map
                                   //kv. first = the label (like "yes" or "no")
                                   //kv.second = how many times it appeared
        if(kv.second > maxCount) { //check to see if the label's count is bigger than the current biggest count
            maxCount = kv.second; //if it is, update maxCount and store the label as the current most common label
            majLabel = kv.first;
        }
    }
    return majLabel; //return the label that appeard the most number of times
}

Node* buildTree(const vector<vector<string>>& data, vector<int> featureIndices, 
                int labelIndex, const string& metric, const vector<string>& attributeNames,
                int depth = 0, int maxDepth = 8) { //recursive function that builds the decision tree
    
    if(data.empty()) {   //Base Case 1: If there is no data
                        //Create a leaf node and label it as unknown
        Node* leaf = new Node(); 
        leaf->isLeaf = true;
        leaf->label = "unknown";
        return leaf;
    }
    
    //Base Case 2: All labels are the same
    string firstLabel = data[0][labelIndex]; //take the label from the first row and store it (almost like a reference label to compare everything else against)
    bool allSame = true;  //assumes all labels are the same (this assumption is tested)
    for(const auto& row : data) { //loop through every row in the dataset
        if(row[labelIndex] != firstLabel) { //compare the current row's label with the first label
            allSame = false;                //if they are different, then set the boolean variable to false
            break;
        }
    }
    if(allSame) {
        Node* leaf = new Node();
        leaf->isLeaf = true;
        leaf->label = firstLabel;
        return leaf;
    }
    
    //Base Case 3: No features left or the maximum depth has been reached
    if(featureIndices.empty() || depth >= maxDepth) { //if no  more features to split or tree is too deep then create a leaf node and label it with the most common class
        Node* leaf = new Node(); 
        leaf->isLeaf = true;
        leaf->label = majorityLabel(data, labelIndex);
        return leaf;
    }
    
    double bestScore = -1e9; 
    int bestFeature = -1; //sentinel
    
    for(int f : featureIndices) { //loop over each candidate feature index f in featureIndices
        double score = 0.0; //variable to hold the metric value for the feature f
        
        if(metric == "gini") { //if the chosen metric is gini,call giniSplit which returns the weighted gini impurity after splitting on feature f
            score = -giniSplit(data, f, labelIndex); //lower giniIndex is better so we negate it
        }
        else if(metric == "info") { //if the chosen metric is info, call the informationGain
            score = informationGain(data, f, labelIndex);
        }
        else if(metric == "gain") { //if the metric is gain, call the gainRatio
            score = gainRatio(data, f, labelIndex);
        }
        
        if(score > bestScore) { //after computing the score for the current feature, check to see if score > bestScore
                                //if so, update the bestScore and set bestFeature to the current feature index f
            bestScore = score;
            bestFeature = f;
        }
    }
    
    if(bestFeature == -1) { //check if no feature was chosen (this can happen if featureIndicies was empty or scores were not better than the initial bestScore)
        Node* leaf = new Node();    // if true, then create a leaf node and set its label to the majority label
        leaf->isLeaf = true;
        leaf->label = majorityLabel(data, labelIndex);
        return leaf;
    }
    
    Node* node = new Node(); //create a new node and store the featureIndex and featureName
    node->isLeaf = false;
    node->featureIndex = bestFeature;
    node->featureName = attributeNames[bestFeature];
    
    map<string, vector<vector<string>>> subsets; //map to store the feature value and the vector of rows that have that feature value
    for(const auto& row : data) { //loop through all rows and append each row into the corresponding bucket
        subsets[row[bestFeature]].push_back(row);
    }
    
    vector<int> remainingFeatures; //new list to ensure that the same feature is not reused down the branch
    for(int f : featureIndices) {
        if(f != bestFeature) {
            remainingFeatures.push_back(f);
        }
    }
    
    for(auto& kv : subsets) { //loop throuh subsets and recursively call buildTree 
        node->children[kv.first] = buildTree(kv.second, remainingFeatures, 
                                             labelIndex, metric, attributeNames, depth + 1, maxDepth);
    }
    
    return node;
}

string predict(Node* node, const vector<string>& sample) { //function that classifies one row of data
    if(node->isLeaf) { //if the current node is a leaf node then there are no more decisions to make and just reutnr its label
        return node->label;
    }
    
    string featureValue = sample[node->featureIndex]; //string that retrieves the value from the intput sample at the feature this node splits on
    
    if(node->children.find(featureValue) != node->children.end()) { //check tosee if the tree has a child branch for that feature value
        return predict(node->children[featureValue], sample);
    }
    
    map<string, int> labelCounts; //map to count how often each label appears among the children
    for(auto& pair : node->children) { //loop through all children of this node and check to see if the child is a leaf node
                                        //if it is a leaf node, simply increase the count for that child's label
        if(pair.second->isLeaf) {
            labelCounts[pair.second->label]++;
        }
    }
    
    string bestLabel; //string to store the most common label
    int maxCount = 0; //int to store the highest count
    for(auto& pair : labelCounts) { //loop throuh over the label coutns and pick the label with the largest frequency
        if(pair.second > maxCount) { 
            maxCount = pair.second;
            bestLabel = pair.first;
        }
    }
    
    if (bestLabel.empty()) { //if no label was found (meaning that no leaf children existed) simply return uknown
         return "unknown";
    } else {
         return bestLabel;  //else return the most common label
    }
}

double calculateAccuracy(Node* tree, const vector<vector<string>>& data, int labelIndex) { //function to measure how accruate the decision tree is
    int correct = 0; //counter to store how many predictions are correct
    for(const auto& row : data) {  //loop through every row in the dataset
        string pred = predict(tree, row); //call prediction funciton and store the predicited label in the string variable pred
        if(pred == row[labelIndex]) { //compare the predicited label to the actual true label from the dataset
            correct++;  //if they match, increase tehcorrect counter
        }
    }
    return (double)correct / data.size() * 100.0; //calculate the final accuracy and then multiply by 100% to get a percentage
}

int main() {
    cout << "Decision Tree Classifier\n";
    cout << "Available Datasets:\n";
    cout << "  1. Thyroid Dataset\n";
    cout << "  2. Adult\n";
    cout << "  3. Mushroom\n";
    cout << "  4. Car Evaluation\n";
    cout << "  5. Nursery\n";
    cout << "  6. Letter Recognition\n";
    cout << "  7. Chess (KR vs KP)\n";
    cout << "  8. Pen-Based Recognition\n";
    cout << "  9. Tic-Tac-Toe\n";
    cout << " 10. Test Dataset (Play Tennis)\n";
    cout << " 11. Student Data\n";
    cout << "Enter dataset number (1-10): ";
    
    int choice; 
    cin >> choice;

    string inputFile;
    vector<string> attributeNames;
    string labelName;
    
    switch (choice) { 
        case 1:
            inputFile = "Thyroid_Diff.csv";
            attributeNames = {"Age","Gender", "Smoking","Hx_Smoking","Hx_Radiotherapy","Thyroid_Function","Physical_examination","Adenopathy","Pathology","Focality","Risk", "T", "N", "M","Stage", "Response", "Target"};
            labelName = "Target";
            break;
        case 2:
            inputFile = "adult.data";
            attributeNames = {
                "age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week", "native-country",
                "income"
            };
            labelName = "income";
            break;
        case 3:
            inputFile = "agaricus-lepiota.data";
            attributeNames = {
                "class", "cap-shape","cap-surface","cap-color","bruises","odor",
                "gill-attachment","gill-spacing","gill-size","gill-color",
                "stalk-shape","stalk-root","stalk-surface-above-ring",
                "stalk-surface-below-ring","stalk-color-above-ring",
                "stalk-color-below-ring","veil-type","veil-color","ring-number",
                "ring-type","spore-print-color","population","habitat"
            };
            labelName = "class";
            break;
        case 4:
            inputFile = "car.data";
            attributeNames = {"buying", "maint", "doors", "persons", "lug_boot", "safety", "evaluation"};
            labelName = "evaluation";
            break;
        case 5:
            inputFile = "nursery.data";
            attributeNames = {
                "parents","has_nurs","form","children","housing",
                "finance","social","health","evaluation"
            };
            labelName = "evaluation";
            break;
        case 6:
            inputFile = "letter-recognition.data";
            attributeNames = {
                "letter", "x-box", "y-box","width","high","onpix","x-bar",
                "y-bar","x2bar","y2bar","xybar","x2ybr","xy2br",
                "x-ege","xegvy","y-ege","yegvx"
            };
            labelName = "letter";
            break;
        case 7:
            inputFile = "krkopt.data";
            attributeNames = {
                "white-king-file", "white-king-rank", "white-rook-file",
                "white-rook-rank", "black-king-file", "black-king-rank", "outcome"
            };
            labelName = "outcome";
            break;
        case 8:
            inputFile = "pen_based.data";
            attributeNames = {
                "x1","x2","x3","x4","x5","x6","x7","x8","x9","x10",
                "x11","x12","x13","x14","x15","x16","digit"
            };
            labelName = "digit";
            break;
        case 9:
            inputFile = "tic-tac-toe.data";
            attributeNames = {
                "top-left", "top-middle", "top-right",
                "middle-left", "middle-middle", "middle-right",
                "bottom-left", "bottom-middle", "bottom-right",
                "outcome"
            };
            labelName = "outcome";
            break;
        case 10:
            inputFile = "test.data";
            attributeNames = {"Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"};
            labelName = "PlayTennis";
            break;
        case 11:
            inputFile = "students.data";
            attributeNames = {
            "marital_status",
            "application_mode",
            "application_order",
            "course",
            "daytime_evening_attendance",
            "previous_qualification",
            "previous_qualification_grade",
            "nationality",
            "mothers_qualification",
            "fathers_qualification",
            "mothers_occupation",
            "fathers_occupation",
            "admission_grade",
            "displaced",
            "educational_special_needs",
            "debtor",
            "tuition_fees_up_to_date",
            "gender",
            "scholarship_holder",
            "age_at_enrollment",
            "international",
            "curricular_units_1st_sem_credited",
            "curricular_units_1st_sem_enrolled",
            "curricular_units_1st_sem_evaluations",
            "curricular_units_1st_sem_approved",
            "curricular_units_1st_sem_grade",
            "curricular_units_1st_sem_without_evaluations",
            "curricular_units_2nd_sem_credited",
            "curricular_units_2nd_sem_enrolled",
            "curricular_units_2nd_sem_evaluations",
            "curricular_units_2nd_sem_approved",
            "curricular_units_2nd_sem_grade",
            "curricular_units_2nd_sem_without_evaluations",
            "unemployment_rate",
            "inflation_rate",
            "gdp",
            "target"       
          };
            labelName = "target";
    break;
        default:
            cerr << "Invalid choice\n";
            return 1;
    }

    // Find label index from attribute names
    int labelIndex = -1;
    for(int i = 0; i < attributeNames.size(); i++) {
        if(attributeNames[i] == labelName) {
            labelIndex = i;
            break;
        }
    }
    
    if(labelIndex == -1) {
        cerr << "Label '" << labelName << "' not found in attributes!\n";
        return 1;
    }

    cout << "\nSelect metric:\n";
    cout << "1. Gini Index\n";
    cout << "2. Information Gain\n";
    cout << "3. Gain Ratio\n";
    cout << "Enter choice (1-3): ";
    
    int metricChoice;
    cin >> metricChoice;
    
    string metric;
    if(metricChoice == 1) 
        metric = "gini";
    else if(metricChoice == 2) 
        metric = "info";
    else if(metricChoice == 3) 
        metric = "gain";
    else { 
        cerr << "Invalid metric\n"; 
        return 1; 
    }

    cout << "\nLoading data from: " << inputFile << endl;

    vector<vector<string>> data = loadDataFile(inputFile);
    vector<vector<string>> trainData, testData;
    splitData(data, trainData, testData);   // 70% train / 30% test
            
    if(data.empty()) {
        cerr << "No data loaded!\n";
        return 1;
    }
        
    cout << "Loaded " << data.size() << " samples with " << data[0].size() << " attributes\n";
    cout << "Target attribute: " << labelName << " (column " << labelIndex << ")\n";

    // Display attribute names
    cout << "\nAttributes:\n";
    for(int i = 0; i < attributeNames.size(); i++) {
        if(i == labelIndex) {
            cout << "  [" << i << "] " << attributeNames[i] << " (TARGET)\n";
        } else {
            cout << "  [" << i << "] " << attributeNames[i] << "\n";
        }
    }

    vector<int> featureIndices;
    for(int i = 0; i < (int)data[0].size(); i++) {
        if(i != labelIndex) {
            featureIndices.push_back(i);
        }
    }
    auto start = chrono::high_resolution_clock::now();

    Node* tree = buildTree(trainData, featureIndices, labelIndex, metric, attributeNames);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    double testAcc  = calculateAccuracy(tree, testData, labelIndex);

    cout << "\nTree Building Time: " << duration.count() / 1000.0 << " seconds" << endl;
    cout << "Validation Accuracy: " << testAcc << "%\n";
    

    ofstream outFile("predictions.txt");
    if(!outFile) {
        cerr << "Error opening predictions.txt for writing\n";
    } else {
        outFile << "SampleID,Actual,Predicted\n";
        for(int i = 0; i < (int)data.size(); i++) {
            string pred = predict(tree, data[i]);
            string actual = data[i][labelIndex];
            outFile << (i+1) << "," << actual << "," << pred << "\n";
        }
        outFile.close();
        cout << "\nAll predictions saved to predictions.txt\n";
    }
    delete tree;
    return 0;
}