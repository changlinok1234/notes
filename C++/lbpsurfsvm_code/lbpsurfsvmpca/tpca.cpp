#include "tpca.h"


Mat tpca(Mat& featuredata,Mat& datamatrix )
{
     /*----------PCA降维-------------------------------*/
 PCA pca(featuredata, cv::Mat(),PCA::DATA_AS_ROW, 0.7);
 Mat vec = pca.eigenvectors.clone();
 Mat pmean=pca.mean.clone();
 Mat result;//( 39, 16384, CV_32FC1 );
 Mat pcaresult;
 PCAProject(datamatrix,pmean,vec,result);
 //PCABackProject(result,pmean,vec,reconstructed);
 pcaresult=result.clone();
 return pcaresult;
}
