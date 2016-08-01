//
//  classifierSVM.cpp
//  hogSvm
//
//  Created by wangruchen on 15/7/13.
//  Copyright (c) 2015年 wangruchen. All rights reserved.
//

#include "classifierSVM.h"

//void classifierSVM(Mat& trainData, Mat& labelsTrain, vector<string> trainImgName, vector<string> testImgName, string kernelFunction, Mat& resultTrain, Mat& resultTest)
void classifierSVM(Mat& trainDataLBPTrain1, Mat& trainDataBOWTrain1, Mat& labelsTrain, vector<string> trainImgName, vector<string> testImgName, string kernelFunction, Mat& resultTrain, Mat& resultTest)
{
//    Mat trainFeatureData(0,1764, CV_32FC1);
//    Mat testFeatureData(0,1764, CV_32FC1);

    string predictTrainFeatureVector="./result/PredictTrain-Features-SUFT.txt";
    string predictTestFeatureVector="./result/PredictTest-Features-SUFT.txt";
    stringstream predictTrainDataTxt;
    stringstream predictTestDataTxt;
    Mat trainDataLBP;//(0,16384, CV_32FC1);
    Mat trainDataBow;//(0,50, CV_32FC1);
    Mat testDataLBP;//(0,16384, CV_32FC1);
    Mat testDataBow;//(0,50, CV_32FC1);

    Mat trainBOW;
    Mat trainLBP;
    /*----------PCA降维-------------------------------*/
    trainLBP=tpca(trainDataLBPTrain1,trainDataLBPTrain1);
    cout<<"trainDataLBPpca"<<trainLBP.size()<<endl;
    /*----------PCA降维-------------------------------*/
    trainBOW=tpca(trainDataBOWTrain1,trainDataBOWTrain1);
    cout<<"trainDatapca"<<trainBOW.size()<<endl;
    
    //特征矩阵转置
    Mat dataMatrix;
    trainLBP=trainLBP.t();
    trainBOW=trainBOW.t();
    dataMatrix.push_back(trainLBP);
    dataMatrix.push_back(trainBOW);
    dataMatrix=dataMatrix.t();
    cout <<"train " << dataMatrix.size()<< endl;

    //SVM::Params params;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);  //没有RBF么？惩罚系数C在哪里？
   //params.svmType = SVM::C_SVC;//C_SVC用于n类分类问题
    //if( kernelFunction == "LINEAR" )
    //params.kernelType = SVM::LINEAR;
    //if( kernelFunction == "RBF" )
    //params.kernelType = SVM::RBF;
    svm->train( dataMatrix , ROW_SAMPLE , labelsTrain );


    /*----------------Predict traindata------------------*/
//    Mat resultTrain;   // 输出分类结果
//    trainFeatureData=extractFeature(trainImgName);
//    WriteData(predictTrainFeatureVector, trainFeatureData);
    
// Mat  trainDataLBPTrain2=LBPextract(trainImgName);
// Mat  trainDataBow1=extractFeature(trainImgName);
//  /*----------PCA降维-------------------------------*/
//  cout <<"LBP " <<trainDataLBPTrain2.size()<< endl;
//      trainDataLBP=tpca(trainDataLBPTrain2,trainDataLBPTrain2);
//       cout <<"LBPpca1 " << trainDataLBP.size()<< endl;
//       cout <<"BOW"<< trainDataBow1.size()<<endl;
//     trainDataBow=tpca(trainDataBow1,trainDataBow1);
//     cout <<"BOWpca1"<< trainDataBow.size()<<endl;
//
//
//    Mat dataMatrix1;
//    trainDataLBP=trainDataLBP.t();
//    trainDataBow=trainDataBow.t();
//    dataMatrix1.push_back(trainDataLBP);
//    dataMatrix1.push_back(trainDataBow);
//    dataMatrix1=dataMatrix1.t();
//     cout <<"train1 " << dataMatrix1.size()<< endl;
//
//    predictTrainDataTxt<<"/home/changlin/program/codeblocks/LBPSURFSVM_correct_灰度图pca/result/PredictTrain-SUFT-SVM-"<<kernelFunction<<".txt";
    svm->predict(dataMatrix, resultTrain);//这里先改成trainData试一下,原本是dataMatrix
    WriteData(predictTrainDataTxt.str(), resultTrain);

    /*----------------Predict testdata-------------------*/
//    Mat resultTest;   // 输出分类结果
//    testFeatureData=extractFeature(testImgName);
  Mat  testDataLBP1=LBPextract(testImgName);
     /*----------PCA降维-------------------------------*/
     cout <<"LBP test "<< testDataLBP1.size()<<endl;
     testDataLBP=tpca(trainDataLBPTrain1,testDataLBP1);
       cout <<"LBP test pca"<< testDataLBP.size()<<endl;

     Mat testDataBow1=extractFeature(testImgName);

    /*----------PCA降维-------------------------------*/
    cout<<"BOW test"<< testDataBow1.size()<<endl;
    testDataBow=tpca(trainDataBOWTrain1,testDataBow1);
       cout<<"BOW test"<< testDataBow.size()<<endl;


    Mat dataMatrixTest;
    testDataLBP=testDataLBP.t();
    testDataBow=testDataBow.t();
    dataMatrixTest.push_back(testDataLBP);
    dataMatrixTest.push_back(testDataBow);
    dataMatrixTest=dataMatrixTest.t();
    cout <<"test " << dataMatrixTest.size()<< endl;

    predictTestDataTxt<<"./result/PredictTest-SUFT-SVM-"<<kernelFunction<<".txt";
    svm->predict(dataMatrixTest, resultTest);
    WriteData(predictTestDataTxt.str(), resultTest);
}
