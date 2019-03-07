# Introduction
This is a Convolutional Neural Network that is able to run on general laptop. You can train your own dataset as long as you define your dataset using ```{'images':[image_1, image_2,...,image_n], 'labels': [label_1,label_2,...,label_n]```. The program is wriiten in python and on Pytorch, and thus it is easy to modified and accomodate to an indiviaul's need. For more detail about the model, please check ```ConvNet.py```.

# ConvNet Architecture
The model has the following simple architecture:
```conv1->ReLu->pool->conv2->ReLu->FC1->Relu->FC2->ReLu```
It is shallow, so you should be able to run on both GPU and CPU.

# Dataset
For demostraction I use [```ORL_faces_dataset```](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html), which look like below

![](Example-images-from-the-ORL-facial-database.png?raw=true)

# Run the model
Make sure you have installed ```pytorch``` and ```numpy```. Simply run ```python3 Main.py```. You should see the prompt showing the loss and accuracy of test dataset
```python
cuda:0
start training

[4,     1] loss: 13.775
Accuracy of the network on the test images: 0 %
[4,    11] loss: 19.054
Accuracy of the network on the test images: 6 %
[5,     1] loss: 8.145
Accuracy of the network on the test images: 13 %
[5,    11] loss: 8.687
Accuracy of the network on the test images: 31 %
[6,     1] loss: 3.405
Accuracy of the network on the test images: 33 %
[6,    11] loss: 4.249
Accuracy of the network on the test images: 46 %
[7,     1] loss: 1.606
Accuracy of the network on the test images: 62 %
[7,    11] loss: 2.598
Accuracy of the network on the test images: 53 %
[8,     1] loss: 1.426
Accuracy of the network on the test images: 65 %
[8,    11] loss: 1.530
Accuracy of the network on the test images: 82 %
[9,     1] loss: 0.534
Accuracy of the network on the test images: 73 %
[9,    11] loss: 1.239
Accuracy of the network on the test images: 70 %
[10,     1] loss: 0.080
Accuracy of the network on the test images: 81 %
[10,    11] loss: 0.136
Accuracy of the network on the test images: 85 %
Finished Trainning
Accuracy of the network on the test images: 87 %
```

In  ```Main.py```, change the parameter ```visualization``` for trainer.train```(num_epochs=10, visualization=False)``` to ```True```, then you can monitor the image prediction for each epoch.

![](prediction.png?raw=true)

That's it! go nuts!
