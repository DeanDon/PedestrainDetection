# PedestrainDetection
使用faster-rcnn实现的行人检测
基于项目faster-rcnn项目的python接口实现：https://github.com/rbgirshick/py-faster-rcnn.git
使用方法：

1.编译安装faster-rcnn的python接口，代码在：https://github.com/rbgirshick/py-faster-rcnn.git

2.下载训练好的caffe模型，百度云链接为：https://pan.baidu.com/s/1w479QUUAwLBS2AJbc-eXIA，将下载的模型文件放到faster-rcnn文件夹的data/faster_rcnn_models文件夹中

3.将本项目中的文件夹替换安装好的faster-rcnn源码中的文件夹

4.使用tools文件夹下的测试脚本运行demo：python person_detect.py

