猫狗大战
=======

一、项目描述
-------

本项目使用深度学习进行猫或狗的宠物图片分类，具体任务如下：<br>
（1）数据集中有猫、狗两类图像。图片名字已注明猫狗类别，需要编写代码根据图片名字中的"cat"或"dog"为图片添加标签"cat(0)"或"dog(1)"。<br>
（2）选择合适的深度学习网络，编写代码，使用给定数据集，合理设置网络参数并训练得到模型，给出模型准确率指标及分析。训练平台不限，选用的深度学习网络不限。<br>
（3）首先使用200张的小样本数据集进行训练和测试，分析所得结果为什么不好。再使用25000张的数据集进行训练和测试，观察结果并给出分析。<br>

二、如何运行
-------------
按照2.1、2.2、2.3、2.4的步骤运行，可完成项目的相关要求。代码包一共包括：<br>
1.	‘cat_dog_alexnet.mat’<br>
2.	‘cat_dog_googlenet.mat’<br>
3.	‘change_size.mat’<br>
4.	‘GoogleNet_classification_test.mat’<br>
运行环境为matlab2020a，大致原理为基于AlexNet与GoogleNet的迁移学习，配置要求要有较高的GPU显卡以及matlab中GoogleNet与AlexNet的程序包。

2.1数据集手动分类
---------
由于数据集已经按照类型进行命名，故采取将容量为200与25000的数据集拆分成两部分，分别存储于名为“cat”（0）与“dog”（1）的文件夹下，记录存储地址。

2.2图片大小调整
--------
由于AlexNet与GoogleNet分别需要输入227x227x3与224x224x3格式的照片，而原数据集的图片大小格式不一，需采取程序自动调整图片格式。<br>
故运行‘change_size.mat’程序，matlab可自动修改路径下所有图片及文件夹中图片的格式为给定标准。
 ```
imds = imageDatastore('D:\Program Files\Polyspace\R2020a\bin\25000_cat_dog', ... %需要修改图片存放地址
 'IncludeSubfolders',true, ...
'LabelSource','foldernames');
numTrainImages = numel(imds.Labels);%读取数据集数量
for i = 1:numTrainImages %统一图片尺寸大小
s = string(imds.Files(i));
I = imread(s);
I = imresize(I,[227,227]);%AlexNet与GoogleNet分别需要输入[227，227][224,224]
imwrite(I,s);
s %在命令窗口显示修改进程
end
```
2.3模型训练及检验过程
-------
将‘cat_dog_alexnet.mat’、‘cat_dog_googlenet.mat’分别导入matlab，对照注释分别执行。<br>

### AlexNet训练代码及注释

#### 加载数据集

```
clc;clear;
close all
imds = imageDatastore('D:\Program Files\Polyspace\R2020a\bin\25000_cat_dog', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');%随机拆分imds中的文件，前70%放入imdsTrain中，剩余30%翻入imdsValidation中
% splitEachLabel-按比例拆分ImageDatastor标签
% [imds1,imds2] = splitEachLabel(imds,p)
% 可将imds中的图像文件拆分为两个新的数据存储，imds1和imds2.新的数据存储imds1包含每个标签前百分之p个文件，imds2中包含剩余的文件
numTrainImages = numel(imdsTrain.Labels);%numel数组中的元素数目
```
### 加载预训练网络
```
Alexnet_Train = alexnet;
%net.Layers %展示这个网络架构，这个网络有5个卷积层和3个全连接层
inputSize = Alexnet_Train.Layers(1).InputSize;
% 第一层是图像输入层，要求输入图像的尺寸为227*227*3 这里的3是颜色通道的数字替换最后三层
layersTransfer = Alexnet_Train.Layers(1:end-3);
% 预处理网络的最后三层被配置为1000个类。这三层必须针对新的分类问题进行调整
numClasses = numel(categories(imdsTrain.Labels));%数据集中类的数目
layers = [
layersTransfer
fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
softmaxLayer
classificationLayer];
% 通过将最后三个图层替换为完全连接图层，softmax图层和分类输出图层，将图层转移到新的分类任务。根据新的数据指定新的完全连接层的选项。将完全连接层设置为与新数据中的类数大小相同。要在新层中比传输层更快的学习，增加完全连接层的WeightLearnRateFactor 和BiasLearnRateFactor的值
```
### 训练网络
```
%用于数据增强，增加数据量这个网络要求的尺寸是227*227*3，但是在图像存储中的图像有不同的尺寸，使用增强数据存储自动调整训练图像大小。在训练图像中指定额外的增强操作：沿着垂直轴随机翻转训练图像，水平和垂直随机移动30个像素单位。
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
'RandXReflection',true, ...
'RandXTranslation',pixelRange, ...
'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
'DataAugmentation',imageAugmenter);% 自动调整验证图像大小而不进行其他数据增强。使用扩充图像数据存储而不指定其他预处理操作。
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);% 指定训练选项，对于传输学习保留来自预训练网络早期层中的特征。
options = trainingOptions('sgdm', ...
'MiniBatchSize',100, ...
'MaxEpochs',6, ...
'InitialLearnRate',1e-4, ...
'ValidationData',augimdsValidation, ...
'ValidationFrequency',3, ...
'ValidationPatience',Inf, ...
'Verbose',true,...
'Plots','training-progress'...
);
Train = trainNetwork(augimdsTrain,layers,options);
% trainNetwork——训练神经网络进行深度学习
```
### 验证训练好的模型
```
[YPred,scores] = classify(Train,augimdsValidation);% classify——使用经过训练的神经网络对数据进行分类
idx = randperm(numel(imdsValidation.Files),20);% randperm——随机置换随机显示使用训练好的模型进行分类的图片及其标签和概率
figure
for i = 1:20
subplot(5,4,i)
I = readimage(imdsValidation,idx(i));
imshow(I)
label = YPred(idx(i));
title(string(label) + "," + num2str(100*max(scores(idx(i),:)),3) + "%");
end
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
disp(['accuracy:',num2str(accuracy)]); % 输出预测精度结果
```
### 保存训练好的模型
```
save Alexnet_25000 Train;
```
## （2）GoogleNet训练代码及注释
### 加载数据
```
clc;close all;clear;
Location = 'D:\Program Files\Polyspace\R2020a\bin\25000_cat_dog';%这里输入自己的数据集地址
imds = imageDatastore(Location ,... %若使用自己的数据集则改为Location（不加单引号）
'IncludeSubfolders',true,...
'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');%将数据集按7:3的比例分为训练集和测试集加载预训练网络
net = googlenet;
```
### 从训练有素的网络中提取图层，并绘制图层图
```
lgraph = layerGraph(net);%从训练网络中提取layer graph
inputSize = net.Layers(1).InputSize;
```
### 替换最终图层
为了训练Googlenet去分类新的图像，取代网络的最后三层。这三层为'loss3-classifier', 'prob', 和 'output'，包含如何将网络的提取的功能组合为类概率和标签的信息。在层次图中添加三层新层：a fully connected layer, a softmax layer, and a classification output layer 将全连接层设置为同新的数据集中类的数目相同的大小，为了使新层比传输层学习更快，增加全连接层的学习因子。
```
lgraph = removeLayers(lgraph,{'loss3-classifier','prob','output'});
numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
fullyConnectedLayer(numClasses,'Name','fc','weightLearnRateFactor',10,'BiasLearnRateFactor',10)
softmaxLayer('Name','softmax')
classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);%将网络中最后一个传输层（pool5-drop_7x7_s1）连接到新层
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
```
### 冻结初始图层

在训练过程中trainNetwork不会跟新冻结层的参数，因为冻结层的梯度不需要计算，冻结大多数初始层的权重对网络训练加速很重要。如果新的数据集很小，冻结早期网络层也可以防止新的数据集过拟合。
```
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:110) = freezeWeights(layers(1:110));%调用freezeWeights函数，设置开始的110层学习速率为0
lgraph = createLgraphUsingConnections(layers,connections);%调用createLgraphUsingConnections函数，按原始顺序重新连接所有的层。
```
### 训练网络
```
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter(...
'RandXReflection',true,...
'RandXTranslation',pixelRange,...
'RandYTranslation',pixelRange);%对输入数据进行数据加强
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
'DataAugmentation',imageAugmenter);% 自动调整验证图像大小
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);%设置训练参数
options = trainingOptions('sgdm', ...
'MiniBatchSize',64, ... %调整参数
'MaxEpochs',8, ... %调整参数
'InitialLearnRate',1e-4, ...
'ValidationData',augimdsValidation, ...
'ValidationFrequency',3, ... %设置验证频率
'ValidationPatience',Inf, ...
'Verbose',true ,...
'Plots','training-progress'); %显示训练过程,开始训练网络
googlenetTrain = trainNetwork(augimdsTrain,lgraph,options);
```
### 随机显示使用训练好的模型进行分类的图片及其标签和概率
```
[YPred,scores] = classify(googlenetTrain,augimdsValidation);
idx = randperm(numel(imdsValidation.Files),20); %随机挑选20张图片进行检验
figure
for i = 1:20
subplot(5,4,i)
I = readimage(imdsValidation,idx(i));
imshow(I)
label = YPred(idx(i));
title(string(label) + "," + num2str(100*max(scores(idx(i),:)),3) + "%");%显示结果与准确度
end
```
### 对验证图像进行分类
```
[YPred,probs] = classify(googlenetTrain,augimdsValidation);%使用训练好的网络进行分类
accuracy = mean(YPred == imdsValidation.Labels)%计算网络的精确度
```
### 保存训练好的模型
```
save googlenet_25000_1 googlenetTrain;% save x y; 保存训练好的模型y（注意：y为训练的模型，即y = trainNetwork()），取名为x
```
## 2.4模型测试
为检验训练好的模型的优良性，选取适当的测试集，检验分类准确性并画出混淆矩阵。运行’GoogleNet_classification_test.mat’相关代码及注释如下：
### 加载模型
```
clc;close all;clear;
load('-mat','D:\Program Files\Polyspace\R2020a\bin\googlenet_25000_1');
```
### 加载测试集
```
Location = 'D:\Program Files\Polyspace\R2020a\bin\25000_cat_dog';
imds = imageDatastore(Location,'includeSubfolders',true,'LabelSource','foldernames');
inputSize = googlenetTrain.Layers(1).InputSize;
imdstest = augmentedImageDatastore(inputSize(1:2),imds);
tic;
YPred = classify(googlenetTrain,imdstest);%使用训练好的模型对测试集进行分类
disp(['分类所用时间为：',num2str(toc),'秒']);
```
### 显示分类结果，绘制混淆矩阵
```
cat = 'cat';
CAT = numel(YPred,YPred == cat);
disp(['cat = ',num2str(CAT)]);
dog = 'dog';
DOG = numel(YPred,YPred == dog);
disp(['dog = ',num2str(DOG)]);
sum = numel(YPred);
disp(['sum = ',num2str(sum)]);
%求出每个标签对应的分类数量
% numel(A) 返回数组A的数目
% numel(A,x) 返回数组A在x的条件下的数目
```
### 计算精确度
```
YTest = imds.Labels;
accuracy = mean(YPred == YTest);
disp(['accuracy = ',num2str(accuracy)]);
% disp(x) 变量x的值
% num2str(x) 将数值数值转换为表示数字的字符数组
```
### 随机显示测试分类后的图片
```
idx = randperm(numel(imds.Files),16);
figure
for i = 1:16
subplot(4,4,i);
I = readimage(imds,idx(i));
imshow(I);
label = YPred(idx(i));
title(string(label));
end
```
### 绘制混淆矩阵
```
predictLabel = YPred;%通过训练好的模型分类后的标签
actualLabel = YTest;%原始的标签
plotconfusion(actualLabel,predictLabel,'Googlenet');%绘制混淆矩阵
% plotconfusion(targets,outputs);绘制混淆矩阵，使用target（true）和output（predict）标签，将标签指定为分类向量或1到N的形
```
