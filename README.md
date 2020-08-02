## recurrect-perception
This code accompanies my Msci project ["Recurrent Perception For Deformable Object Recogniton"](https://drive.google.com/file/d/1OCmKT6ss_VzrcCSFNrRgx4IoPIJQT3wc/view?usp=sharing)

![Move](/img/cpdataset.png?raw=true "Sample move")  

## Running
1. Clone this repo, create a new Conda environment and install dependencies.
The accompanying dataset will be pulled in `data/` (small enough to fit into git).
```
$ git clone https://github.com/domantasjurkus/recurrent-perception.git
$ conda create --name recurrent-perception --file requirements.txt
```

2. Run the desired train/test script. All runnable scripts are prefixed with `main_`.
Each script will train a model from scratch. No saved models are uploaded since the network should train pretty fast.
The model for each script is denoted by name.
```
python main_singleshot.py
python main_snippet.py
python main_fullvideo.py
python main_sliding_window.py
```

