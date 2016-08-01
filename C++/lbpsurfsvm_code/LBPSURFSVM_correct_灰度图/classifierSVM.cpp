//
//  classifierSVM.cpp
//  hogSvm
//
//  Created by wangruchen on 15/7/13.
//  Copyright (c) 2015年 wangruchen. All rights reserved.
//

#include "classifierSVM.h"

void classifierSVM(Mat& trainData, Mat& labelsTrain, vector<string> trainImgName, vector<string> testImgName, string kernelFunction, Mat& resultTrain, Mat& resultTest)
{
//    Mat trainFeatureData(0,1764, CV_32FC1);
//    Mat testFeatureData(0,1764, CV_32FC1);

    string predictTrainFeatureVector="/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/result/PredictTrain-Features-SUFT.txt";
    string predictTestFeatureVector="/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/result/PredictTest-Features-SUFT.txt";
    stringstream predictTrainDataTxt;
    stringstream predictTestDataTxt;
    Mat trainDataLBP;//(0,16384, CV_32FC1);
    Mat trainDataBow;//(0,50, CV_32FC1);
    Mat testDataLBP;//(0,16384, CV_32FC1);
    Mat testDataBow;//(0,50, CV_32FC1);


    //SVM::Params params;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);  //没有RBF么？惩罚系数C在哪里？
   //params.svmType = SVM::C_SVC;//C_SVC用于n类分类问题
    //if( kernelFunction == "LINEAR" )
    //params.kernelType = SVM::LINEAR;
    //if( kernelFunction == "RBF" )
    //params.kernelType = SVM::RBF;
    svm->train( trainData , ROW_SAMPLE , labelsTrain );


    /*----------------Predict traindata------------------*/
//    Mat resultTrain;   // 输出分类结果
//    trainFeatureData=extractFeature(trainImgName);
//    WriteData(predictTrainFeatureVector, trainFeatureData);
    trainDataLBP=LBPextract(trainImgName);
    trainDataBow=extractFeature(trainImgName);

    Mat dataMatrix;
    trainDataLBP=trainDataLBP.t();
    trainDataBow=trainDataBow.t();
    dataMatrix.push_back(trainDataLBP);
    dataMatrix.push_back(trainDataBow);
    dataMatrix=dataMatrix.t();


    predictTrainDataTxt<<"/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/result/PredictTrain-SUFT-SVM-"<<kernelFunction<<".txt";
    svm->predict(dataMatrix, resultTrain);
    WriteData(predictTrainDataTxt.str(), resultTrain);

    /*----------------Predict testdata-------------------*/
//    Mat resultTest;   // 输出分类结果
//    testFeatureData=extractFeature(testImgName);
    testDataLBP=LBPextract(testImgName);
    testDataBow=extractFeature(testImgName);


    Mat dataMatrixTest;
    testDataLBP=testDataLBP.t();
    testDataBow=testDataBow.t();
    dataMatrixTest.push_back(testDataLBP);
    dataMatrixTest.push_back(testDataBow);
    dataMatrixTest=dataMatrixTest.t();

    predictTestDataTxt<<"/home/changlin/program/codeblocks/lbpsurfsvm_code/LBPSURFSVM_correct_灰度图/result/PredictTest-SUFT-SVM-"<<kernelFunction<<".txt";
    svm->predict(dataMatrixTest, resultTest);
    WriteData(predictTestDataTxt.str(), resultTest);
}
