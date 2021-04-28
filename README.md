# Where can i drive - keras implementation
Keras Implementation of the encoder decoder network "Where can I drive" as
probposed in the [Arxive](https://arxiv.org/abs/2004.07639]) article.

## Getting started
This implementation uses tensorflow keras.
However, no special tensorflow functions are used.

1. Clone this git repository
2. Install `reqirements.txt` file in your preferred environment
```
pip install -r requirements.txt
```

## Train the network

## Predict images
Envoke the prediction via the CLI.

```
usage: predict.py [-h] -i INPUT -e EXTENSION -m MODEL -o OUTPUT [-v {grayscale,heatmap,binary}] [-t THRESHOLD] [-mt] [-p] [--height HEIGHT] [-w WIDTH]

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
