# License-plate-recognition
原始训练集来自大佬博客：https://blog.csdn.net/shadown1ght/article/details/78571187
省份训练集只有['京', '闽', '粤', '苏', '沪', '浙']6个，但是字母和数字是都全的。我在其中加入了现实生活拍摄的部分照片进行扩充，最后把训练集图片导出成了csv，方便进行读取。

目前只能识别蓝色车牌，因为我第一步就是将图片的色彩空间转为HSV色彩空间，然后识别图中的蓝色区域，所以貌似有些车子本身就是蓝色会识别不了。然后还有一点是没有对车牌进行矫正，所以如果图片拍摄角度很歪会影响识别。尽量保证识别的车牌图片清晰，拍摄角度不是很歪，一般可以识别。有些识别不了可能是因为没有定位到或者定位的不准导致识别不准，所以有可能把图片裁剪小一点发现又能识别了，等有时间再改进吧，小伙伴有什么问题或者意见可以提，能star就更好啦，谢谢啦！


![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test0.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test1.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test2.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test3.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test4.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test5.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test6.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test7.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test8.png)
![image](https://github.com/duanshengliu/License-plate-recognition/blob/master/main/some_result_pic/test9.png)
