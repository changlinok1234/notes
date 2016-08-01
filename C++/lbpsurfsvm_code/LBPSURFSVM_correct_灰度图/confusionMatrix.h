//
//  confusionMatrix.h
//  hogSvm
//
//  Created by wangruchen on 15/7/12.
//  Copyright (c) 2015年 wangruchen. All rights reserved.
//

#ifndef __tmp__confusionMatrix__
#define __tmp__confusionMatrix__

#include <opencv2/core.hpp>
#include <fstream>
#include "WriteData.h"

using namespace cv;
using namespace std;

Mat confusionMatrix(Mat labelsTrain, Mat predictTrainResult, int classifyNum, string confusionMatrixTrainName);

#endif /* defined(__hogSvm__confusionMatrix__) */
