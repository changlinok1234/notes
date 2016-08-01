//
//  classifierSVM.h
//  hogSvm
//
//  Created by wangruchen on 15/7/13.
//  Copyright (c) 2015年 wangruchen. All rights reserved.
//

#ifndef __tmp__classifierSVM__
#define __tmp__classifierSVM__

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include "extractFeature.h"
#include "LBPextract.h"
#include "WriteData.h"
#include "tpca.h"
using namespace cv;
using namespace std;
using namespace ml;



void classifierSVM(Mat& trainDataLBP1, Mat& trainData1, Mat& labelsTrain, vector<string> trainImgName, vector<string> testImgName, string kernelFunction, Mat& resultTrain, Mat& resultTest);



#endif /* defined(__hogSvm__classifierSVM__) */
