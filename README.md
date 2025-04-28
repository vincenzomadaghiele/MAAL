# ALL: Autonomous Live looper

The Autonomous Live Looper (ALL) is a co-creative sampler/looper based on a multi-agent logic algorithm and machine listening. The ALL is composed of several agents, each controlling a loop track, which can autonomously decide to sample and play back segments of a live vocal performance by listening to each other. 
The Autonomous Live Looper aims to expands the possibilities for indirect control, interaction, and co-creativity in live looping for improvising musicians. 

<img src="https://github.com/vincenzomadaghiele/ALL-Autonomous-Live-Looper/blob/main/ALL_GUI.png" alt="drawing"  width="80%"/>


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

Open a terminal. Configure the settings of the looper by modifying the file `config.json`; set the audiofile to be used for the offline ALL in the python script. Then run:
```
python3 offlineALL.py
```
This will generate a the corresponding audiotracks and visualizations in a new folder in `./01_output_offline/...`.



## Playing with the Online ALL

#### Python
Open a terminal. Configure the settings of the looper by modifying the file `config.json`. Then run:
```
python3 onlineALL.py
```


## Configuration options
The ALL can be configured by changing the settings in a `./config.json` file. This is a summary of the possible configuration options:


| Settings name | Description | Value range |
| --- | --- | :--: |
| <b>tempo</b>: <i>int</i> |  |  |
| <b>beats_per_loop</b>: <i>int</i> |  |  |
| <b>rhythm_subdivision</b>: <i>int</i> |  |  |
| <b>startup-mode</b>: <i>string</i> |  |  |
| <b>startup-repetition-numBars</b>: <i>int</i> |  |  |
| <b>startup-similarityThreshold</b>: <i>float</i> |  |  |
| <b>startup-firstLoopBar</b>: <i>int</i> |  |  |
| <b>minLoopsRepetition</b>: <i>int</i> |  |  |
| <b>maxLoopsRepetition</b>: <i>int</i> |  |  |
| <b>loopChange-rule</b>: <i>string</i> |  |  |
| <b>looping-rules</b>: <i>list of RuleCombination</i> |  |  |
| <b>RuleCombination</b>: <i>list of Rule</i> |  |  |
| <b>Rule</b>: <i> dict with keys {</i> |  |  |
| <b>&nbsp; &nbsp; rule-name</b>: <i>string</i> |  | rule name from the table <b>Comparison metrics</b> |
| <b>&nbsp; &nbsp; rule-type</b>: <i>string</i> |  | <i>more</i> or <i>less</i> |
| <b>&nbsp; &nbsp; rule-threshold</b>: <i>float</i> |  |  |
| <i>}</i> |  |  |


### Comparison metrics

| Metric name | Descriptors computed | Comparison method |
| --- | --- | :--: |
| <b>Harmonic similarity</b> | Chroma | MSE on sequence |



