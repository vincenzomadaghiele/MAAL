import json
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client

import sequence_descriptors as des
import compare_sequences as comp


# SOUND CONTROLS
sr = 44100
BPM = 120
BEATS_PER_LOOP = 4 # tempo
RHYTHM_SUBDIVISIONS = 16 # bar quantization
N_BAR_SAMPLES = int(1 / (BPM / 60) * sr * BEATS_PER_LOOP) # number of samples in a bar: 1 / BPS * framerate * beats_per_bar
FFT_WINDOW = 1024
FFT_HOP_SIZE = 512
N_FFT_FRAMES = int(N_BAR_SAMPLES / FFT_HOP_SIZE) + 1


# LOOPER CONTROLS
N_LOOPS = 6

N_CHROMA = 12
N_MELBANDS = 40
N_SPECTRALSHAPE = 7
N_LOUDNESS = 2
N_ONSET = 1


# INITIALIZE FEATURE VECTORS
# sum of all other loops vector
chroma_loops = np.zeros((N_LOOPS, N_CHROMA, N_FFT_FRAMES))
melbands_loops = np.zeros((N_LOOPS, N_MELBANDS, N_FFT_FRAMES))
spectralshape_loops = np.zeros((N_LOOPS, N_SPECTRALSHAPE, N_FFT_FRAMES))
loudness_loops = np.zeros((N_LOOPS, N_LOUDNESS, N_FFT_FRAMES))
onsets_loops = [[] for _ in range(N_LOOPS)]
binaryRhythms_loops = [[] for _ in range(N_LOOPS)]
# sequence feature vectors
chroma_sequence = np.zeros((N_CHROMA, N_FFT_FRAMES))
melbands_sequence = np.zeros((N_MELBANDS, N_FFT_FRAMES))
spectralshape_sequence = np.zeros((N_SPECTRALSHAPE, N_FFT_FRAMES))
loudness_sequence = np.zeros((N_LOUDNESS, N_FFT_FRAMES))
onsets_sequence = []
binaryRhythms_sequence = []


# checking features received
N_FEATURES = N_CHROMA + N_MELBANDS + N_SPECTRALSHAPE + N_LOUDNESS + N_ONSET
EXPECTED_NUM_FEATURES = N_FEATURES * (N_LOOPS + 1)
featuresInCounter = 0



def default_handler(address, *args):
    print(f"DEFAULT {address}: {len(args)}")


def liveFeaturesIn_handler(address, *args):

	global featuresInCounter
	global binaryRhythms_sequence, chroma_sequence, melbands_sequence, spectralshape_sequence, loudness_sequence, onsets_sequence
	global binaryRhythms_loops, chroma_loops, melbands_loops, spectralshape_loops, loudness_loops, onsets_loops

	#print(f"{address}: {len(args)}")
	feature_name = address.split('/')[-1].split('-')[0]
	loop_num = int(address.split('/')[2])
	feature_component_num = int(address.split('/')[-1].split('-')[-1])

	# SIMPLE FEATURE RECEIVER
	if loop_num == 1000:
		if feature_name == 'chroma':
			chroma_sequence[feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'spectralshape':
			spectralshape_sequence[feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'melbands':
			melbands_sequence[feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'loudness':
			loudness_sequence[feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'onsets':
			onsets = np.abs(np.array(args))
			# binary rhythm representation
			interval_size = int(N_BAR_SAMPLES / RHYTHM_SUBDIVISIONS)
			binary_rhythm = []
			for i in range(0, N_BAR_SAMPLES, interval_size):
				# if there is a onset in the bar division 1, otherwise 0
				flag = 0
				flag_dynamic = 0
				for onset in onsets:
					if onset > i and onset <= i+interval_size:
						flag = 1
				binary_rhythm.append(flag)
			onsets_sequence = (onsets / FFT_HOP_SIZE).astype(int)
			binaryRhythms_sequence = binary_rhythm
			#print(binary_rhythm)
			featuresInCounter += 1
	else:
		if feature_name == 'chroma':
			chroma_loops[loop_num, feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'spectralshape':
			spectralshape_loops[loop_num, feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'melbands':
			melbands_loops[loop_num, feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'loudness':
			loudness_loops[loop_num, feature_component_num, :] = np.array(args)[:N_FFT_FRAMES]
			featuresInCounter += 1
		elif feature_name == 'onsets':
			onsets = np.abs(np.array(args))
			onsets_loops[loop_num] = (onsets / FFT_HOP_SIZE).astype(int)
			# binary rhythm representation
			interval_size = int(N_BAR_SAMPLES / RHYTHM_SUBDIVISIONS)
			binary_rhythm = []
			for i in range(0, N_BAR_SAMPLES, interval_size):
				# if there is a onset in the bar division 1, otherwise 0
				flag = 0
				flag_dynamic = 0
				for onset in onsets:
					if onset > i and onset <= i+interval_size:
						flag = 1
				binary_rhythm.append(flag)
			binaryRhythms_loops[loop_num] = binary_rhythm
			#print(binary_rhythm)
			featuresInCounter += 1

	# ACT WHEN ALL FEATURES HAVE BEEN RECEIVED
	if featuresInCounter >= EXPECTED_NUM_FEATURES:
		
		featuresInCounter = 0
		print('-'*50)
		# all features received 
		# processing here

		# check all loops
		for i in range(N_LOOPS):

			# process features
			# current sequence features
			onsets_seq = onsets_sequence
			binary_rhythm_seq = binaryRhythms_sequence
			chroma_seq = chroma_sequence[:,:]
			discretechroma_seq = np.array([chroma_seq[:,j] for j in onsets_seq])
			loudness_seq = loudness_sequence[0,:]
			discreteloudness_seq = np.array([loudness_seq[j] for j in onsets_seq])
			centroid_seq = spectralshape_sequence[0,:]
			discretecentroid_seq = np.array([centroid_seq[j] for j in onsets_seq])
			flatness_seq = spectralshape_sequence[5,:]
			discreteflatness_seq = np.array([flatness_seq[j] for j in onsets_seq])

			# sum of loops features
			onsets_sum = onsets_loops[i]
			binary_rhythm_sum = binaryRhythms_loops[i]
			chroma_sum = chroma_loops[i,:,:]
			discretechroma_sum = np.array([chroma_sum[:,j] for j in onsets_sum])
			loudness_sum = loudness_loops[i,0,:]
			discreteloudness_sum = np.array([loudness_sum[j] for j in onsets_sum])
			centroid_sum = spectralshape_loops[i,0,:]
			discretecentroid_sum = np.array([centroid_sum[j] for j in onsets_sum])
			flatness_sum = spectralshape_loops[i,5,:]
			discreteflatness_sum = np.array([flatness_sum[j] for j in onsets_sum])

			# compute comparison metrics
			print('Binary rhythms:')
			binary_comparison_coefficient, rhythm_density_coefficient = comp.compareBinaryRhythms(binary_rhythm_seq, binary_rhythm_sum, RHYTHM_SUBDIVISIONS)

			## SPECTRAL BANDWIDTH
			#print('Spectral bandwidth:')
			#spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient = compareSpectralBandwidth(CQT_bar, CQT_center_of_mass_bar, CQT_var_bar, CQT_sum, CQT_center_of_mass_sum, CQT_var_sum)

			## CHROMA
			print('Chroma:')
			chroma_AE = comp.computeTwodimensionalAE(chroma_seq, chroma_sum)
			_, chroma_continuous_correlation = comp.computeTwodimensionalContinuousCorrelation(chroma_seq, chroma_sum)
			_, chroma_discrete_correlation = comp.computeTwodimensionalDiscreteCorrelation(onsets_seq, discretechroma_seq, onsets_sum, discretechroma_sum)

			## LOUDNESS
			print('Loudness:')
			loudness_MSE = comp.computeSignalsMSE(loudness_seq.reshape(-1,1), loudness_sum.reshape(-1,1))
			_, loudness_continuous_correlation = comp.computeContinuousCorrelation(loudness_seq.reshape(-1,1), loudness_sum.reshape(-1,1))
			_, loudness_discrete_correlation = comp.computeDiscreteCorrelation(onsets_seq, discreteloudness_seq, onsets_sum, discreteloudness_sum)

			## CENTROID
			print('Spectral centroid:')
			centroid_MSE = comp.computeSignalsMSE(centroid_seq.reshape(-1,1), centroid_sum.reshape(-1,1))
			_, centroid_continuous_correlation = comp.computeContinuousCorrelation(centroid_seq.reshape(-1,1), centroid_sum.reshape(-1,1))
			_, centroid_discrete_correlation = comp.computeDiscreteCorrelation(onsets_seq, discretecentroid_seq, onsets_sum, discretecentroid_sum)

			## FLATNESS
			print('Spectral flatness:')
			flatness_MSE = comp.computeSignalsMSE(flatness_seq.reshape(-1,1), flatness_sum.reshape(-1,1))
			_, flatness_continuous_correlation = comp.computeContinuousCorrelation(flatness_seq.reshape(-1,1), flatness_sum.reshape(-1,1))
			_, flatness_discrete_correlation = comp.computeDiscreteCorrelation(onsets_seq, discreteflatness_seq, onsets_sum, discreteflatness_sum)
			print()

			rhythm_metrics = [binary_comparison_coefficient, rhythm_density_coefficient]
			#bandwidth_metrics = [spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient]
			chroma_metrics = [chroma_AE, chroma_continuous_correlation, chroma_discrete_correlation]
			loudness_metrics = [loudness_MSE, loudness_continuous_correlation, loudness_discrete_correlation],
			centroid_metrics = [centroid_MSE, centroid_continuous_correlation, centroid_discrete_correlation]
			flatness_metrics = [flatness_MSE, flatness_continuous_correlation, flatness_discrete_correlation]
			

			# EVALUATE LOOP CANDIDATE BASED ON RULES COMBINATION
			n_rule_components = len(looping_rules[i])
			rules_satisfied = []
			rules_satisfaction_degree = [] # between 0 and 1
			# iterate over rule components
			for rule in looping_rules[i]:
				rule_satisfied = False
				rule_satisfaction_degree = 0
				threshold = rule["rule-threshold"]

				# BINARY RHYTHM SIMILARITY 
				if rule["rule-name"] == "binaryRhythm-similarity":
					if rule["rule-type"] == "major":
						if rhythm_metrics[0] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(rhythm_metrics[0] - threshold)
					elif rule["rule-type"] == "minor":
						if rhythm_metrics[0] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - rhythm_metrics[0])

				# BINARY RHYTHM DENSITY
				elif rule["rule-name"] == "binaryRhythmDensity-similarity":
					if rule["rule-type"] == "major":
						if rhythm_metrics[1] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(rhythm_metrics[1] - threshold)
					elif rule["rule-type"] == "minor":
						if rhythm_metrics[1] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - rhythm_metrics[1])

				# CHROMA ABSOLUTE DIFFERENCE
				elif rule["rule-name"] == "chroma-absoluteDifference":
					if rule["rule-type"] == "major":
						if chroma_metrics[0] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(chroma_metrics[0] - threshold)
					elif rule["rule-type"] == "minor":
						if chroma_metrics[0] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - chroma_metrics[0])

				# CHROMA CONTINUOUS CORRELATION
				elif rule["rule-name"] == "chroma-continuousCorrelation":
					if rule["rule-type"] == "major":
						if chroma_metrics[1] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(chroma_metrics[1] - threshold)
					elif rule["rule-type"] == "minor":
						if chroma_metrics[1] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - chroma_metrics[1])

				# CHROMA DISCRETE CORRELATION
				elif rule["rule-name"] == "chroma-discreteCorrelation":
					if rule["rule-type"] == "major":
						if chroma_metrics[2] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(chroma_metrics[2] - threshold)
					elif rule["rule-type"] == "minor":
						if chroma_metrics[2] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - chroma_metrics[2])


				# LOUDNESS MEAN SQUARE DIFFERENCE
				elif rule["rule-name"] == "loudness-meanSquareDifference":
					if rule["rule-type"] == "major":
						if loudness_metrics[0] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(loudness_metrics[0] - threshold)
					elif rule["rule-type"] == "minor":
						if loudness_metrics[0] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - loudness_metrics[0])

				# LOUDNESS CONTINUOUS CORRELATION
				elif rule["rule-name"] == "loudness-continuousCorrelation":
					if rule["rule-type"] == "major":
						if loudness_metrics[1] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(loudness_metrics[1] - threshold)
					elif rule["rule-type"] == "minor":
						if loudness_metrics[1] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - loudness_metrics[1])

				# LOUDNESS DISCRETE CORRELATION
				elif rule["rule-name"] == "loudness-discreteCorrelation":
					if rule["rule-type"] == "major":
						if loudness_metrics[2] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(loudness_metrics[2] - threshold)
					elif rule["rule-type"] == "minor":
						if loudness_metrics[2] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - loudness_metrics[2])
				

				# CENTROID MEAN SQUARE DIFFERENCE
				elif rule["rule-name"] == "centroid-meanSquareDifference":
					if rule["rule-type"] == "major":
						if centroid_metrics[0] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(centroid_metrics[0] - threshold)
					elif rule["rule-type"] == "minor":
						if centroid_metrics[0] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - centroid_metrics[0])

				# CENTROID CONTINUOUS CORRELATION
				elif rule["rule-name"] == "centroid-continuousCorrelation":
					if rule["rule-type"] == "major":
						if centroid_metrics[1] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(centroid_metrics[1] - threshold)
					elif rule["rule-type"] == "minor":
						if centroid_metrics[1] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - centroid_metrics[1])

				# CENTROID DISCRETE CORRELATION
				elif rule["rule-name"] == "centroid-discreteCorrelation":
					if rule["rule-type"] == "major":
						if centroid_metrics[2] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(centroid_metrics[2] - threshold)
					elif rule["rule-type"] == "minor":
						if centroid_metrics[2] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - centroid_metrics[2])


				# FLATNESS MEAN SQUARE DIFFERENCE
				elif rule["rule-name"] == "flatness-meanSquareDifference":
					if rule["rule-type"] == "major":
						if flatness_metrics[0] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(flatness_metrics[0] - threshold)
					elif rule["rule-type"] == "minor":
						if flatness_metrics[0] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - flatness_metrics[0])

				# FLATNESS CONTINUOUS CORRELATION
				elif rule["rule-name"] == "flatness-continuousCorrelation":
					if rule["rule-type"] == "major":
						if flatness_metrics[1] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(flatness_metrics[1] - threshold)
					elif rule["rule-type"] == "minor":
						if flatness_metrics[1] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - flatness_metrics[1])

				# FLATNESS DISCRETE CORRELATION
				elif rule["rule-name"] == "flatness-discreteCorrelation":
					if rule["rule-type"] == "major":
						if flatness_metrics[2] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(flatness_metrics[2] - threshold)
					elif rule["rule-type"] == "minor":
						if flatness_metrics[2] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - flatness_metrics[2])
				
				'''
				# SPECTRAL ENERGY OVERLAP
				elif rule["rule-name"] == "spectralEnergy-overlap":
					if rule["rule-type"] == "major":
						if bandwidth_metrics[0] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(bandwidth_metrics[0] - threshold)
					elif rule["rule-type"] == "minor":
						if bandwidth_metrics[0] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - bandwidth_metrics[0])

				# SPECTRAL ENERGY DIFFERENCE
				elif rule["rule-name"] == "spectralEnergy-difference":
					if rule["rule-type"] == "major":
						if bandwidth_metrics[1] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(bandwidth_metrics[1] - threshold)
					elif rule["rule-type"] == "minor":
						if bandwidth_metrics[1] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - bandwidth_metrics[1])
				'''

				rules_satisfied.append(rule_satisfied)
				rules_satisfaction_degree.append(rule_satisfaction_degree)


			# CHECK IF LOOP SHOULD BE CHANGED
			if bars_loop_persisted[i] >= IMPATIENCE_THR:
				# CHECK IF LOOP CONDIDATE SATISFIES ALL RULES
				if all(rules_satisfied):
					if LOOP_CHANGE_RULE == "newer":
						print('-'*50)
						print(f'BAR SELECTED FOR LOOP {i+1}')
						print('-'*50)
						bars_loop_persisted[i] = 0
						active_loops[i] = True
						selected_loops_satisfaction_degrees[i] = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
						client.send_message("/loopdecision/loop", str(i))
						break
					elif LOOP_CHANGE_RULE == "better":
						if sum(rules_satisfaction_degree)/len(rules_satisfaction_degree) >= selected_loops_satisfaction_degrees[i]:
							print('-'*50)
							print(f'BAR SELECTED FOR LOOP {i+1}')
							print('-'*50)
							bars_loop_persisted[i] = 0
							active_loops[i] = True
							selected_loops_satisfaction_degrees[i] = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
							client.send_message("/loopdecision/loop", str(i))
							break

			# CHECK IF LOOP SHOULD BE DROPPED
			if bars_loop_persisted[i] >= BORED_THR:
				print(f'DROPPING LOOP {i}')
				bars_loop_persisted[i] = 0
				selected_loops_satisfaction_degrees[i] = 0
				active_loops[i] = False
				client.send_message("/loopdecision/drop", str(i))
			else: 
				bars_loop_persisted[i] += 1



if __name__ == '__main__': 


	# LOAD LOOPER CONFIGURATION FILE
	with open('config.json', 'r') as file:
		config = json.load(file)
	print(config)
	looping_rules = config["looping-rules"]
	IMPATIENCE_THR = config["minLoopsRepetition"]
	BORED_THR = config["maxLoopsRepetition"]
	LOOP_CHANGE_RULE = config["loopChange-rule"] # better OR newer
	# checks on config file
	if LOOP_CHANGE_RULE != "newer" and LOOP_CHANGE_RULE != "better":
		LOOP_CHANGE_RULE = "newer"
	STARTUP_MODE = config["startup-mode"]
	STARTUP_SIMILARITY_THR = config["startup-similarityThreshold"]
	N_BARS_STARTUP = config["startup-numBars"]


	# INITIALIZE LOOPER
	# looper state features
	bars_loop_persisted = np.zeros((N_LOOPS)).tolist()
	selected_loops_satisfaction_degrees = [0 for _ in range(N_LOOPS)]
	active_loops = [False for _ in range(N_LOOPS)]

	
	# LOAD PURE DATA LOOPER


	## OSC SERVER

	# network parameters
	ip = "127.0.0.1" # localhost
	port_snd = 6667 # send port to PD
	port_rcv = 6666 # receive port from PD

	# define dispatcher
	dispatcher = Dispatcher()
	dispatcher.map("/features/*", liveFeaturesIn_handler)
	dispatcher.set_default_handler(default_handler)

	# define client
	client = udp_client.SimpleUDPClient(ip, port_snd)

	# define server
	server = BlockingOSCUDPServer((ip, port_rcv), dispatcher)
	server.serve_forever()  # Blocks forever

