# ALL: Autonomous Live looper

The Autonomous Live Looper (ALL) is a co-creative sampler/looper based on a multi-agent logic algorithm and machine listening. The ALL is composed of several agents, each controlling a loop track, which can autonomously decide to sample and play back segments of a live vocal performance by listening to each other. 
The Autonomous Live Looper aims to expands the possibilities for indirect control, interaction, and co-creativity in live looping for improvising musicians. 

<p style="text-align: center;">
<img src="https://github.com/vincenzomadaghiele/ALL-Autonomous-Live-Looper/blob/main/ALL_GUI.png" alt="drawing"  width="80%"/>	
</p>


## Installing dependencies

#### Python
Download and install anaconda [here](https://puredata.info/downloads).
Open a terminal and run the following instructions to install the dependencies:
```
conda env create -f looper-environment.yml
conda activate looper
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
python3 offlineALL.py
```
This will generate a the corresponding audiotracks and visualizations in a new folder in `./01_output_offline/...`.


## Playing with the Online ALL

#### Python
Navigate in the terminal to the directory of the server, decide the settings of the looper by modifying the file `config.json`. Then run:
```
python3 onlineALL.py
```


