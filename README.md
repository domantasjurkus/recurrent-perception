## recurrect-perception
This code accompanies my Msci project "Recurrent Perception For Deformable Object Recogniton"

## Running
1. Clone this repo, create a new Conda environment and install dependencies:
```
$ git clone ...
$ conda create --name recurrent-perception --file requirements.txt
```

2. Download and unzip the Continuous Perception dataset (cleaned and prepped for training) into the project root:
http://www.mediafire.com/file/0cfhjg16yphubud/recurrent-perception-dataset.zip/file

3. Run the desired train/test script. All runnable scripts are prefixed with `main_`.
Each script will train a model from scratch. No saved models are uploaded since the network should train pretty fast.
The model for each script is denoted by name:
```
python main_singleshot.py
python main_snippet.py
python main_fullvideo.py
python main_sliding_window.py
```

