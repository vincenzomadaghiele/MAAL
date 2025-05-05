# ALL: Autonomous Live Looper

The Autonomous Live Looper (ALL) is a co-creative sampler/looper based on a multi-agent logic algorithm and machine listening. The ALL is composed of several agents, each controlling a loop track, which can autonomously decide to sample and play back segments of a live vocal performance by listening to each other. 
The Autonomous Live Looper aims to expands the possibilities for indirect control, interaction, and co-creativity in live looping for improvising musicians. 

<img src="https://github.com/vincenzomadaghiele/ALL-Autonomous-Live-Looper/blob/main/ALL_GUI.png" alt="drawing"  width="100%"/>


More information about the project is in the paper:
> Vincenzo Madaghiele, Stefano Fasciani, Tejaswinee Kelkar, Çagri Erdem.
> [**ALL: Autonomous Live Looper for improvised co-creation of musical structures**]().
> In _Proceedings of AI and Music Creativity Conference (AIMC) 2025_, 10-12 September 2025, Bruxelles (BE).


## Installation

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

The `else` library for PD is used for GUI objects in PD, it can be installed by typing `else` in the deken externals manager (`Help -> find externals`) and clicking on `install`.


## Offline ALL

Open a terminal. Configure the settings of the looper by modifying a configuration file like `config.json` in this repository; set the audiofile to be used for the offline ALL in the python script. Then run:
```
python3 offlineALL.py --SOUNDFILE_FILEPATH <path/to/soundfile.wav> --CONFIG_FILEPAHT <path/to/configfile.json> --OUTPUT_DIR_PATH <path/to/outputdir>
```
This will generate a the corresponding audiotracks and visualizations in a new folder with the same name as the soudfile in `<path/to/outputdir>`.



## Online ALL

Open a terminal. Configure the settings of the looper by modifying a configuration file like `config.json` in this repository. Then run:
```
python3 onlineALL.py --CONFIG_FILEPAHT <path/to/configfile.json>
```
This python script will load and set up a PD patch as well.


## Configuration options
The ALL can be configured by changing the settings in a `./config.json` file. This is a list of the possible configuration options:

| Settings name | Description | Value range |
| --- | --- | :--: |
| <b>tempo</b>: <i>int</i> | Tempo determining the duration of a beat in beats per minute (BPM). | any |
| <b>beats_per_loop</b>: <i>int</i> | Number of beats corresponding to the duration of a loop. | any |
| <b>rhythm_subdivision</b>: <i>int</i> | Quantization of a loop for rhythm analysis. | any |
| <b>startup-mode</b>: <i>string</i> | Mode for selection of first loop. | <i>repetition</i> or <i>user-set</i> |
| <b>startup-repetition-numBars</b>: <i>int</i> | Number of consecutive repetitions at for <i>repetition</i> startup mode. | any |
| <b>startup-similarityThreshold</b>: <i>float</i> | Similarity threshold for <i>repetition</i> startup mode. | [0,1] |
| <b>startup-firstLoopBar</b>: <i>int</i> | Number of first loop to be selected in <i>user-set</i> startup mode. | any |
| <b>minLoopsRepetition</b>: <i>int</i> | Minimum number of repetitions for a loop before it can be dropped. | any |
| <b>maxLoopsRepetition</b>: <i>int</i> | Maximum number of repetitions for a loop after which it is dropped. | any |
| <b>loopChange-rule</b>: <i>string</i> | Rule for changing the content of a loop track. | <i>better</i> or <i>newer</i>  |
| <b>looping-rules</b>: <i>list of RuleCombination</i> | A list of rule combinations for each loop track. The ALL infers the number of loop tracks from the number of elements in this list. It is composed of RuleCombination lists. | as many as the number of tracks |
| <b>RuleCombination</b>: <i>list of Rule</i> | A list of rules used for a loop track, composed of many Rule objects. | any |
| <b>Rule</b>: <i> dict with keys {</i> | A rule object. It is a dictionary with three elements. |  |
| <b>&nbsp; &nbsp; rule-name</b>: <i>string</i> | The name of the rule. | rule name from the table [comparison metrics](#comparison-metrics) |
| <b>&nbsp; &nbsp; rule-type</b>: <i>string</i> | The type of rule to be used for comparison. | <i>more</i> or <i>less</i> |
| <b>&nbsp; &nbsp; rule-threshold</b>: <i>float</i> | The threshold used for comparison. | [0,1] |
| <i>}</i> |  |  |


### Comparison Metrics

The all works by comparing sound segments according to specific sequence-level comparison functions inspired by musical criteria. Comparison functions evaluate the similarity of two sequences according to musical criteria, returning a similarity index between 0 and 1. Comparison functions can be combined into rules to assign musical functions to each loop track. This is a list of the possible comparison functions: 

| Metric name | Descriptors computed | Comparison method |
| :---: | --- | -- |
| Harmonic similarity| Chroma | MSE on sequence |
| Harmonic movement - C | Chroma | PCC on sequence |
| Harmonic movement - D | Chroma, Onset | PCC at onsets |
| Melodic similarity | Centroid | MSE on sequence |
| Melodic trajectory - C | Centroid | PCC on sequence |
| Melodic trajectory - D | Centroid, Onset |  PCC at onsets |
| Dynamic similarity | RMS | MSE on sequence |
| Dynamic changes - C | RMS | PCC on sequence |
| Dynamic changes - D | RMS, Onset |  PCC at onsets |
| Timbral similarity | Flatness | MSE on sequence |
| Timbral evolution - C | Flatness | PCC on sequence |
| Timbral evolution - D | Flatness, Onset |  PCC at onsets |
| Global spectral overlap | Mel Spectrum  | Difference between averages |
| Frequency range overlap |  Mel Spectrum  | Overlap of estimated bandwidths |
| Rhythmic similarity | Binary Rhythm, Onset | Hamming distance | 
| Rhythmic density | Binary Rhythm, Onset | MSE on quantized sequence | 



## Cite
```
@inproceedings{madaghiele2024RLimpro,
  author    = {Madaghiele, Vincenzo and Fasciani, Stefano and Kelkar, Tejaswinee and Erdem, Çagri},
  title     = {{ALL: Autonomous Live Looper for improvised co-creation of musical structures}},
  booktitle = {Proceedings of AI and Music Creativity Conference (AIMC)},
  year      = {2025},
  month     = {09},
  address   = {Bruxellses (BE)}
}
```


