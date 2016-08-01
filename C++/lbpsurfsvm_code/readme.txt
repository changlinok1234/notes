LBPSURFSVM_correct_灰度图，是用来提取特征，没有进行pca降维就分类的程序，

lbpsurfsvmpca,是提取特征后，经过pca降维再分类的程序，使用分类数为３的原数据，在pca降维值为0.5时可以取得一个好的识别效果,5类那组不晓得为啥降到多少识别率最高是88%

在终端执行 bash ./KaggleLabel.sh 可以生成Test_Set.txt、Training_Set.txt里面的路径
