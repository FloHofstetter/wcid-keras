# Where can i drive - keras implementation
Keras Implementation of the encoder decoder network _Where can I drive_ as
probposed in the [ArXiv.org](https://arxiv.org/abs/2004.07639]) article.

## Getting started
This implementation uses tensorflow keras.
However, no special tensorflow functions are used.

1. Clone this git repository
2. Install `reqirements.txt` file in your preferred environment
```
pip install -r requirements.txt
```

## Train the network
Envoke training via CLI:

```
usage: train.py [-h] [-o OUTPUT] [-ep EPOCHS] [-l LEARNING_RATE] [-b BATCH_SIZE]
                [-g GPU] [--height HEIGHT] [-w WIDTH]
                train_images train_masks val_images val_masks extension

positional arguments:
  train_images          Path to the folder containing RGB training images.
  train_masks           Path to the folder containing training masks.
  val_images            Path to the folder containing RGB validation images.
  val_masks             Path to the folder containing validation masks.
  extension             Name of the file extension. For example: '-e jpg''.
  background_pth        Path to direcotry contatining images used for background switch augmentation.  
                        For example: '/path_to/random_images/'.


optional arguments:
  -h, --help
        show this help message and exit

  -o OUTPUT, --output OUTPUT
        Path to the folder where to store model path and other training artifacts.

  -ep EPOCHS, --epochs EPOCHS
        Training epochs.

  -l LEARNING_RATE, --learning_rate LEARNING_RATE
        Learning rate for the adam solver.

  -b BATCH_SIZE, --batch_size BATCH_SIZE
        Batch size of training and validation.

  -g GPU, --gpu GPU
        Select the GPU id to train on.

  --height HEIGHT
        Height of the neural network input layer.

  -w WIDTH, --width WIDTH
        Width of the neural network input layer.

  --horizontal_flip HORIZONTAL_FLIP
        Probability of flipping image in training set. Default is 0.5.

  --brightness_contrast BRIGHTNESS_CONTRAST
        Probability of applying random brightness contrast on image in training set. Default is 0.2.

  --rotation ROTATION   
        Probability of applying random rotation on image in training set. Default is 0.9.

  --motion_blur MOTION_BLUR
        Probability of applying motion blur on image in training set. Default is 0.1.

  --background_swap BACKGROUND_SWAP
        Probability of applying background swap on image in training set. Default is 0.9.
```

## Predict images
Envoke prediction via CLI:

```
usage: predict.py [-h] [-v {grayscale,heatmap,binary}] [-t THRESHOLD]
                  [-mt] [-p] [--height HEIGHT] [-w WIDTH]
                  input extension model output

positional arguments:
  input                 Path to the folder with the RGB images to be processed.
  extension             Name of the file extension. For example: <-e jpg>.
  model                 Path to the architecture/model file.
  output                Path to folder in which the segmented images are to be stored.

optional arguments:
  -h, --help            
        show this help message and exit
  
  -v {grayscale,heatmap,binary}, --vistype {grayscale,heatmap,binary}
        Visualisation type. Default is grayscale.
  
  -t THRESHOLD, --threshold THRESHOLD
        Threshold for binary classification. Default is 0.5.
  
  -mt, --multiple-thresholds
        Store all thresholds from 0-10 in 1, 10-100 in 10, 90-100 in 1 steps.
  
  -p, --progress        
        Show progress bar on stdout.
  
  --height HEIGHT
        Height of the output image.
  
  -w WIDTH, --width WIDTH
        Width of the output image.
```
