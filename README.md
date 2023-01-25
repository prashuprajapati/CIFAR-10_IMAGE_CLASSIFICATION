# CIFAR-10_IMAGE_CLASSIFICATION
The CIFAR-10 dataset is 
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
he CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. It's kind of famous in the computer vision community and it is often used as (toy) benchmark. It's a nice dataset to play with. It's a bit like MNIST, but there are cats and dogs and frogs! And there are colors too!

Despite its fame, I did not find any easy plug-and-play wrapper around it. Of course, there are wrappers to CIFAR-10 in most deep learning frameworks (TensorFlow, PyTorch) but you know I usually don't want to get into a whole deep learning framework just to play with 32x32 cat images. So here's why. And yes, I also wanted to have some fun learning pathlib.



