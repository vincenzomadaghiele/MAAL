# ALL: Autonomous Live looper

The Autonomous Live Looper is a co-creative software system that delegates the choice of the musical segments to be looped during live performances to a set of rule-based algorithmic agents. Each loop track is associated with a machine-listening-based agent that utilizes rules to continuously compare segments concurrently being looped with newly played segments by a live improviser, that can be added or replaced in the loop track. Each loop track's agent listens to the other ongoing loops and makes decisions based on the compatibility of their sonic or musical properties, characterizing the entire system as a multi-agent network capable of generating emergent behaviors. The type of compatibility is determined by selecting, for each loop track, one or a combination of sonic rules from a pool we provide, along with associated thresholds to express arbitrary degrees of musical similarity or dissimilarity. 
The Autonomous Live Looper aims to expands the possibilities for indirect control, interaction, and co-creativity in live looping for improvising musicians. 


## Installing dependencies

#### Python
Download and install anaconda [here](https://puredata.info/downloads).
Open a terminal and run the following instructions to install the dependencies:
```
conda env create -f looper-environment.yml
conda activate looper
```

Alternativeley, you can install dependencies one by one by running the following instructions:
```
conda create env -n loops
conda activate loops
pip install python-osc
pip install numpy
pip install matplotlib
pip install librosa
pip install scipy
```

#### Pure Data
Download Pure Data (PD) [here](https://puredata.info/downloads).
Download the Flucoma library for PD following the instructions [here](https://learn.flucoma.org/installation/pd/). 

The `zexy` library for PD is used for OSC communication between python and PD, it can be installed by typing `zexy` in the deken externals manager (`Help -> find externals`) and clicking on `install`.

The `iem_tab` library for PD is used for buffer operations in PD, it can be installed by typing `iem_tab` in the deken externals manager (`Help -> find externals`) and clicking on `install`.

## Generating tracks with the Offline ALL

#### Python
Navigate in the terminal to the directory of the server, decide the settings of the looper by modifying the file `config.json`; set the audiofile to be used for the offline ALL in the python script. Then run:
```
python3 generate_offlineALL.py
```
This will generate a the corresponding audiotracks and visualizations in a new folder in `./01_output_offline/...`.

## Playing with the Online ALL

#### Python
Navigate in the terminal to the directory of the server, decide the settings of the looper by modifying the file `config.json`. Then run:
```
python3 onlineALLclass.py
```

#### Pure Data
Open the PD patch `./02_ALL_PD/_main.pd` and follow the numbered instructions in the patch.


