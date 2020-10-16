# Vegetables and Fruit Classification using DCNN (Re-trained)
This repository is to explain the code and Convolutional Neural Network that I use for my bachelor thesis project.
Note that all of the model in here is retrained and there are some changes made to the model, which will be explained later on.

## Introduction
As far as we know, Convolutional Neural Network (CNN) are already do a good job with classification problem.
And thus I wanted to implement a CNN model toward a fridge to automatically classify the items that being put inside it. 
It also can be used towards the Automatic Online Inventory System.
So, in regards to that, I will make a CNN Model that can be run in tablet, or phones while limiting the models to predict some classes as follows:
<br>
|      |     |       |      |     |    |     |      |         |
|------|-----|-------|------|-----|----|-----|------|---------|
|fruits|apple|avocado|banana|grape|kiwi|mango|orange|pineapple|
|vegetables|carrot|garlic|tomato||||||


## Dataset
The dataset are just an images taken from [Google](https://images.google.com).
<br>
<img src="https://github.com/Xinihiko/retrained-FruVe11/blob/master/others/Example1.png" width="45%" height="45%"/>
<img src="https://github.com/Xinihiko/retrained-FruVe11/blob/master/others/Example2.png" width="45%" height="45%"/>
<br>
Originally, the images used is consists of 2 datasets originially, but I made the third dataset that consist of the "best" class from accuracy from the two dataset combined.
The dataset consist of:
<table class="tg">
<thead>
  <tr>
    <th class="tg-wp8o" rowspan="2">Class</th>
    <th class="tg-73oq" colspan="2">Dataset 1</th>
    <th class="tg-73oq" colspan="2">Dataset 2</th>
    <th class="tg-73oq" colspan="2">Dataset 3</th>
  </tr>
  <tr>
    <td class="tg-73oq">Train</td>
    <td class="tg-73oq">Test</td>
    <td class="tg-73oq">Train</td>
    <td class="tg-73oq">Test</td>
    <td class="tg-73oq">Train<br></td>
    <td class="tg-73oq">Test</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-73oq">Apple</td>
    <td class="tg-73oq">58<br></td>
    <td class="tg-73oq">5</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">58</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-73oq">Avocado</td>
    <td class="tg-73oq">50</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-auyb"><span style="color:#333">8</span></td>
    <td class="tg-73oq">50</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-73oq">Banana</td>
    <td class="tg-73oq">36</td>
    <td class="tg-73oq">6</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">36</td>
    <td class="tg-73oq">9</td>
  </tr>
  <tr>
    <td class="tg-73oq">Carrot</td>
    <td class="tg-73oq">27</td>
    <td class="tg-73oq">6</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">27</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-73oq">Garlic<br></td>
    <td class="tg-73oq">36</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">36</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-73oq">Grape<br></td>
    <td class="tg-73oq">41</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-73oq">Kiwi</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">5</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-73oq">Mango</td>
    <td class="tg-73oq">45</td>
    <td class="tg-73oq">6</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">45<br></td>
    <td class="tg-73oq">10</td>
  </tr>
  <tr>
    <td class="tg-73oq">Orange</td>
    <td class="tg-73oq">35</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">9</td>
  </tr>
  <tr>
    <td class="tg-73oq">Pineapple</td>
    <td class="tg-73oq">34</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-73oq">Tomato</td>
    <td class="tg-73oq">51</td>
    <td class="tg-73oq">7</td>
    <td class="tg-73oq">60</td>
    <td class="tg-73oq">8</td>
    <td class="tg-73oq">51</td>
    <td class="tg-73oq">8</td>
  </tr>
  <tr>
    <td class="tg-mcqj">Total</td>
    <td class="tg-mcqj">473</td>
    <td class="tg-mcqj">75</td>
    <td class="tg-mcqj">660<br></td>
    <td class="tg-mcqj">88</td>
    <td class="tg-mcqj">543</td>
    <td class="tg-mcqj">92</td>
  </tr>
</tbody>
</table>
But in this re-trained project, I changed the dataset by combining the train and test images from dataset 3, and ensuring that there is no duplicate images by hashing all the images and remove the images with the same hash keys.
Then I loaded the images as an array, preprocess (normalize and convert the label to one hot) and save it into npz file.
The result is as follows:<br><br>

categories|apple|avocado|banana|carrot|garlic|grape|kiwi|mango|orange|pineapple|tomato|total
----------|-----|-------|------|------|------|-----|----|-----|------|---------|------|-----
images|78|74|73|76|75|74|76|71|77|76|74|824

From those images, I take 11% (91 images) of all data to be the validation or test dataset, while uses the rest for training.

## References for the Model
To create this model I've read some paper that considered as the state of the art CNN models such as:
- Residual Network (ResNet),
- You Only Look Once (YOLO) backbone, DarkNet-19 and DarkNet-53,
- InceptionNet or GoogLeNet,
- Densely Connected Convolutional Networks (DenseNet),
<img src="https://github.com/Xinihiko/retrained-FruVe11/blob/master/others/ResNet, Yolo, Inception.png" width="100%" />
And from those networks, I have tried to implement the ideas written from those papers such as the blocks, or the structure.

## Model Creating
To create the model, firstly I goes from basic CNN which stacks ConvNet and MaxPool, train for 250 epoch and evaluate it, then implement the block and changing the structure or the kernel sizes, channels, and other things.
But, starting from the 11th or 12th model, I've started to implement a input splitter which consist of blocks that taken 1 input and split it into 2 inputs, and split it again to 3 inputs.
The implemented input splitter is a component from Proposed Architecture, where every strand (result from splitter) have its own blocks used, and later on it will be concatenated and finally goes to Softmax layer.
After reaching the 30th model, I evaluate all of the model and take the best 7 from them all.

![Splitter](https://github.com/Xinihiko/retrained-FruVe11/blob/master/others/Splitter.png)

And the result is as follows:
| | Base Architecture | Depth | Layer | Information |
|-|-------------------|-------|-------|-------------|
CNN_2|Standard CNN|37|54|Residual: Conv, Conv|
CNN_3|Standard CNN|40|57|Residual: Stacked Bottleneck, Conv|
CNN_5|Standard CNN|40|61|Residual: Stacked Bottleneck, Bottleneck|
CNN_11|Proposed Architecture|40|125|Factorization layer|
CNN_14|Proposed Architecture|37|122|No Factorization layer|
CNN_29|Proposed Architecture|44|151|3 Block each strand|
CNN_30|Proposed Architecture|48|173|4 Block each strand|

## Training
Previously, I used Data Augmentation with settings as:
1. Image are rescaled or normalized by dividing it by 255,
2. Image may be rotated randomly within -20 to 20 degrees,
3. Image may be flipped horizontally,
4. Image may be shifted from -10% to 10% from the original size horizontally or vertically.

But now, I changed some settings to:
1. Image may be zoomed 5% from the original image,
2. Image may be rotated randomly within -90 to 90 degrees,
3. Image may be flipped horizontally and vertically,
4. Image may be shifted from -10% to 10% from the original size horizontally or vertically.
5. Image may be sheared to 15 degrees.

## Results
The result after training can be seen either as confusion matrix or F1 Scores and accuracy.

### Confusion Matrix
#### CNN 2
![CNN_2](https://github.com/Xinihiko/retrained-FruVe11/blob/master/Confusion%20Matrix%20CNN_2.png)
#### CNN 3
![CNN_3](https://github.com/Xinihiko/retrained-FruVe11/blob/master/Confusion%20Matrix%20CNN_3.png)
#### CNN 5
![CNN_5](https://github.com/Xinihiko/retrained-FruVe11/blob/master/Confusion%20Matrix%20CNN_5.png)
#### CNN 11
![CNN_11](https://github.com/Xinihiko/retrained-FruVe11/blob/master/Confusion%20Matrix%20CNN_11.png)
#### CNN 14
![CNN_14](https://github.com/Xinihiko/retrained-FruVe11/blob/master/Confusion%20Matrix%20CNN_14.png)
#### CNN 29
![CNN_29](https://github.com/Xinihiko/retrained-FruVe11/blob/master/Confusion%20Matrix%20CNN_29.png)
#### CNN 30
![CNN_30](https://github.com/Xinihiko/retrained-FruVe11/blob/master/Confusion%20Matrix%20CNN_30.png)

### Accuracy
|Model|CNN 2|CNN 3|CNN 5|CNN 11|CNN 14|CNN 29|CNN 30|
|-----|-----|-----|-----|------|------|------|------|
|Accuracy|76|81|84|85|80|89|87|

<br>
Previous result showed that CNN 30 is the best from them all, but it is caused by the mistakes in CNN 29, whereas the previous CNN 29 model doesn't have any flatten layer.
CNN 29 previously have an output of (None, 1, 1, 11) rather than (None, 11). 
And as we can see, it shows that CNN 30 is the best if you didn't consider the CNN 29.
This caused some errors that previously have been taken care of by using sparse categorical crossentropy rather than categorical crossentropy.
That ends up showing wrong result, where this time, I added a flatten layer to make the output to be (None, 11).
<br>


## References
* He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. 2016 IEEE Converence on Computer Vision and Patter Recognition (CVPR), 1-6. doi: 10.1109/CVPR.2016.90
* Huang, G., Liu, Z., Maaten, L., V., D., Weinberger, K., Q.(2017) Densely connected convolutional networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2261 - 2269. doi: 10.1109/CVPR.2017.243
* Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv: 1502.03167v3
* Li, B., & He, Y. (2018). An improved ResNet based on the adjustable shortcut connections. IEEE Access, 6, 18967 - 18974. doi: 10.1109/ACCESS.2018.2814605
* Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., . . . Rabinovich, A. (2015). Going Deeper with convolutions. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1-9. doi: 10.1109/CVPR.2015.7298594
* Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2818 - 2826. doi: 10.1109/CVPR.2016.308




