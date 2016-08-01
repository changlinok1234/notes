#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include "pathAndLabeltrain.h"
#include "pathAndLabeltest.h"
#include "LBPextract.h"
#include "extractFeature.h"
#include "WriteData.h"
#include "confusionMatrix.h"
#include "classifierSVM.h"
#include "tpca.h"
#include <fstream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;

//extern void pathAndLabeltest(string dataName, Mat& labels, vector<string>& imgName);

int main(int argc, const char * argv[]) {
    int classifyNum=5;
    string imgTrainDataName="./Training_Set.txt";
    string imgTestDataName="./Test_Set.txt";
    string trainFeatureVector="./result/Train-Features-SUFT.txt";
    string trainFeatureLBPVector="./result/Train-Features-LBP.txt";
    Mat labelsTrain(0,1,CV_32FC1);
    Mat labelsTest(0,1,CV_32FC1);
    Mat trainDataLBP;//(0,16384, CV_32FC1);
    Mat trainData;//(0,50, CV_32FC1);  //原来130那里是50
    //16384怎么来的？



    vector<string> trainImgName;
    vector<string> testImgName;

    pathAndLabeltrain(imgTrainDataName, labelsTrain, trainImgName);
    pathAndLabeltest(imgTestDataName, labelsTest, testImgName);

    //  Mat trainDataLBP(0,16384, CV_32FC1); //16384怎么来的？

    /*---------------Train Feature Vector LBP-----------------*/
Mat trainDataLBP1=LBPextract(trainImgName);
    WriteData(trainFeatureVector, trainDataLBP1);
    

    /*---------------Train Feature Vector SURF-----------------*/
  Mat  trainData1=extractFeature(trainImgName);
    WriteData(trainFeatureVector, trainData1);

    

    /*-----------------------Classifier-----------------------*/
    string kernelFunction;
    for (int i=0; i<2; i++) {
        if (i==0) {
            kernelFunction="LINEAR";  //线性可分
        }
        else if (i==1) {
            kernelFunction="RBF";  //线性不可分
        }
        Mat predictTrainResult,predictTestResult;   // 输出分类结果
//        classifierSVM(dataMatrix, labelsTrain, trainImgName, testImgName, kernelFunction, predictTrainResult, predictTestResult);
        classifierSVM(trainDataLBP1, trainData1, labelsTrain, trainImgName, testImgName, kernelFunction, predictTrainResult, predictTestResult);
       /*cout << labelsTrain.size() << endl;
        cout << labelsTest.size() << endl;
        cout << predictTestResult.size() << endl;*/

        /*----------------- Confusion Matrix-----------------*/
        for (int j=0; j<2; j++) {
            stringstream confusionMatrixTrainName;
            stringstream confusionMatrixTestName;
            if (j==0) {
                confusionMatrixTrainName<<"./result/Train-CM-SUFT-SVM-"<<kernelFunction<<".txt";
                confusionMatrix(labelsTrain, predictTrainResult, classifyNum, confusionMatrixTrainName.str());
            }
            else if (j==1) {
                confusionMatrixTestName<<"./result/Test-CM-SUFT-SVM-"<<kernelFunction<<".txt";
                confusionMatrix(labelsTest, predictTestResult, classifyNum, confusionMatrixTestName.str());
            }
        }
    }
    return 0;
}
