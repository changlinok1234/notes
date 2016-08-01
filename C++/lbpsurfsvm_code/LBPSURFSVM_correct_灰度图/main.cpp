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
#include <fstream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;

//extern void pathAndLabeltest(string dataName, Mat& labels, vector<string>& imgName);

int main(int argc, const char * argv[]) {
    int classifyNum=5;
    string imgTrainDataName="/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/Training_Set.txt";
    string imgTestDataName="/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/Test_Set.txt";
    string trainFeatureVector="/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/result/Train-Features-SUFT.txt";

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
    trainDataLBP=LBPextract(trainImgName);
    WriteData(trainFeatureVector, trainDataLBP);

    /*---------------Train Feature Vector SURF-----------------*/
    trainData=extractFeature(trainImgName);
    WriteData(trainFeatureVector, trainData);

   /*---------------LBP&SURF矩阵合并-trainDataLBP&trainData-------------------------*/
  //查看特征矩阵大小，这里其实不用查看，因为大小已经给定了
  /*int a,b,c,d;
  a=trainDataLBP.rows;
  b=trainDataLBP.cols;
  c=trainData.rows;
  d=trainData.cols;
  ofstream outfile;
outfile.open("myfile.txt"); //myfile.bat是存放数据的文件名
if(outfile.is_open())
{
outfile<<a<<endl; //message是程序中处理的数据
outfile<<b<<endl; //message是程序中处理的数据
outfile<<c<<endl; //message是程序中处理的数据
outfile<<d<<endl; //message是程序中处理的数据
outfile.close();
}
else
{
cout<<"不能打开文件!"<<endl;
}*/

  //特征矩阵转置
    Mat dataMatrix;
      cout<<trainData.size()<<endl;
      cout<<trainData.rows<<endl;
    trainDataLBP=trainDataLBP.t();
    trainData=trainData.t();

    dataMatrix.push_back(trainDataLBP);
    dataMatrix.push_back(trainData);
    dataMatrix=dataMatrix.t();
     cout<<dataMatrix.size()<<endl;
    cout<<dataMatrix.rows<<endl;


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
        classifierSVM(dataMatrix, labelsTrain, trainImgName, testImgName, kernelFunction, predictTrainResult, predictTestResult);
        //cout << labelsTrain.size() << endl;
        //cout << labelsTest.size() << endl;
        cout << predictTestResult.size() << endl;

        /*----------------- Confusion Matrix-----------------*/
        for (int j=0; j<2; j++) {
            stringstream confusionMatrixTrainName;
            stringstream confusionMatrixTestName;
            if (j==0) {
                confusionMatrixTrainName<<"/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/result/Train-CM-SUFT-SVM-"<<kernelFunction<<".txt";
                confusionMatrix(labelsTrain, predictTrainResult, classifyNum, confusionMatrixTrainName.str());
            }
            else if (j==1) {
                confusionMatrixTestName<<"/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/result/Test-CM-SUFT-SVM-"<<kernelFunction<<".txt";
                confusionMatrix(labelsTest, predictTestResult, classifyNum, confusionMatrixTestName.str());
            }
        }
    }
    return 0;
}
