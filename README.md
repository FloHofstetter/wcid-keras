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
usage: train.py [-h] -ti TRAIN_IMAGES -tm TRAIN_MASKS -vi VAL_IMAGES -vm VAL_MASKS -ex EXTENSION  
                [-o OUTPUT] [-ep EPOCHS] [-l LEARNING_RATE] [-b BATCH_SIZE] [-g GPU] 
                [--height HEIGHT] [-w WIDTH]
                
optional arguments:
  -h, --help            
        show this help message and exit

  -ti TRAIN_IMAGES, --train_images TRAIN_IMAGES
        Path to the folder containing RGB training images.

  -tm TRAIN_MASKS, --train_masks TRAIN_MASKS
        Path to the folder containing training masks.

  -vi VAL_IMAGES, --val_images VAL_IMAGES
        Path to the folder containing RGB validation images.

  -vm VAL_MASKS, --val_masks VAL_MASKS
        Path to the folder containing validation masks.

  -ex EXTENSION, --extension EXTENSION
        Name of the file extension. For example: <-e jpg>.

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
```

## Predict images
Envoke prediction via CLI:

```
usage: predict.py [-h] -i INPUT -e EXTENSION -m MODEL -o OUTPUT  
                  [-v {grayscale,heatmap,binary}] [-t THRESHOLD]
                  [-mt] [-p] [--height HEIGHT] [-w WIDTH]

optional arguments:
  -h, --help
        show this help message and exit
  
  -i INPUT, --input INPUT
        Path to the folder with the RGB images to be processed.
  
  -e EXTENSION, --extension EXTENSION
        Name of the file extension. For example: <-e jpg>.
  
  -m MODEL, --model MODEL
        Path to the architecture/model file.
  
  -o OUTPUT, --output OUTPUT
        Path to folder in which the segmented images are to be stored.
  
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
