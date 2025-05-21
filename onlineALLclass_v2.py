import json
import os
import threading
import librosa
import numpy as np
import numpy.lib.recfunctions
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import matplotlib.pyplot as plt

# PD executables
macos_pd_executable = '/Applications/Pd-0.55-2.app/Contents/Resources/bin/pd' # on mac
ubuntu_pd_executable = '/usr/bin/pd' # on linux


class AutonomousLooperOnline():

	def __init__(self, 
				config_filepath='./config.json',
				sr=44100,
				fft_window=1024,
				fft_hopSize=512,
				ip = "127.0.0.1", # localhost
				port_snd = 6667, # send port to PD
				port_rcv = 6666, # receive port from PD
				UBUNTU=False,
				verbose=1 # 0, 1, 2
				):

		print()
		print('Initializing Autonomous Looper online')
		print('-'*50)
		print()

		# LOAD LOOPER PROPERTIES FROM CONFIGURATION FILE
		with open(config_filepath, 'r') as file:
			config = json.load(file)
		print('Configuration options:')
		print(json.dumps(config, indent=4))
		self.config = config
		self.looping_rules = config["looping-rules"]
		self.MIN_LOOPS_REPETITION = config["minLoopsRepetition"] # minimun number of times a loop is repeated
		self.MAX_LOOPS_REPETITION = config["maxLoopsRepetition"] # maximum number of times a loop is repeated
		self.LOOP_CHANGE_RULE = config["loopChange-rule"] # "better" OR "newer"
		if self.LOOP_CHANGE_RULE != "newer" and self.LOOP_CHANGE_RULE != "better":
			LOOP_CHANGE_RULE = "newer"
		self.STARTUP_MODE = config["startup-mode"] # "user-set" OR "repetition"
		if self.STARTUP_MODE != "user-set" and self.STARTUP_MODE != "repetition":
			self.STARTUP_MODE = "user-set"
		self.STARTUP_SIMILARITY_THR = config["startup-similarityThreshold"]
		self.N_BARS_STARTUP = config["startup-repetition-numBars"] # only relevant if STARTUP_MODE != "user-set"
		self.STARTUP_LOOP_BAR_N = config["startup-firstLoopBar"] # first loop set by users
		self.TEMPO = config["tempo"] # bpm
		self.BEATS_PER_LOOP = config["beats_per_loop"] # T_u
		self.MIN_BEATS_PER_LOOP = config["min_loop_beats"] # T_l
		self.RHYTHM_SUBDIVISIONS = config["rhythm_subdivision"] # bar quantization
		self.verbose = verbose

		# DEFINE ALL NAMES OF LOOPING CRITERIA
		# the rules in the list should be in the same order as the vector that computes them		
		self.RULE_NAMES = ["Harmonic similarity", "Harmonic movement - C", "Harmonic movement - D",
							"Melodic similarity", "Melodic trajectory - C", "Melodic trajectory - D",
							"Dynamic similarity", "Dynamic changes - C", "Dynamic changes - D",
							"Timbral similarity", "Timbral evolution - C", "Timbral evolution - D",
							"Global spectral overlap", "Frequency range overlap",
							"Rhythmic similarity", "Rhythmic density",
							"Harmonic function similarity", "Harmonic function transitions - C", "Harmonic function transitions - D"]

		self.N_LOOPS = len(self.looping_rules)

		# Execute PD
		#if not UBUNTU:
			#pd_looper_path = './02_ALL_PD_2/_main.pd'
			#command = macos_pd_executable + f' -send "; N_LOOPS {self.N_LOOPS}; BPM {self.TEMPO}; fftsize {self.FFT_WINDOW}; hopsize {self.FFT_HOP_SIZE}; " ' + pd_looper_path
			#os.system(command)
			#thread.start_new_thread(os.system, (command,))
			# start all programs
			#process = subprocess.Popen(command, shell=True) 
			#process.wait()

		# PD CONFIGURATIONS
		self.sr = sr
		self.FFT_WINDOW = fft_window
		self.FFT_HOP_SIZE = fft_hopSize
		self.N_BAR_SAMPLES = int(1 / (self.TEMPO / 60) * self.sr * self.BEATS_PER_LOOP) # number of samples in a bar: 1 / BPS * framerate * beats_per_bar
		self.N_FFT_FRAMES = int(self.N_BAR_SAMPLES / self.FFT_HOP_SIZE) + 1
		self.frames_per_beat = int(self.N_FFT_FRAMES / self.BEATS_PER_LOOP)

		# NON MODIFIABLE DYNAMICALLY
		self.N_CHROMA = 12
		self.N_MELBANDS = 40
		self.N_SPECTRALSHAPE = 7
		self.N_LOUDNESS = 2
		self.N_PITCH = 2
		self.N_ONSET = 1
		self.N_TONNETZ = 6 # COMPUTED IN PYTHON

		# INITIALIZE FEATURE VECTORS
		# sum of all other loops vector
		self.chroma_loops = np.zeros((self.N_LOOPS, self.N_CHROMA, self.N_FFT_FRAMES))
		self.tonnetz_loops = np.zeros((self.N_LOOPS, self.N_TONNETZ, self.N_FFT_FRAMES))
		self.melbands_loops = np.zeros((self.N_LOOPS, self.N_MELBANDS, self.N_FFT_FRAMES))
		self.spectralshape_loops = np.zeros((self.N_LOOPS, self.N_SPECTRALSHAPE, self.N_FFT_FRAMES))
		self.loudness_loops = np.zeros((self.N_LOOPS, self.N_LOUDNESS, self.N_FFT_FRAMES))
		self.pitch_loops = np.zeros((self.N_LOOPS, self.N_PITCH, self.N_FFT_FRAMES))
		self.onsets_loops = [[] for _ in range(self.N_LOOPS)]
		self.binaryRhythms_loops = [[] for _ in range(self.N_LOOPS)]
		# sequence feature vectors
		self.chroma_sequence = np.zeros((self.N_CHROMA, self.N_FFT_FRAMES))
		self.tonnetz_sequence = np.zeros((self.N_TONNETZ, self.N_FFT_FRAMES))
		self.melbands_sequence = np.zeros((self.N_MELBANDS, self.N_FFT_FRAMES))
		self.spectralshape_sequence = np.zeros((self.N_SPECTRALSHAPE, self.N_FFT_FRAMES))
		self.loudness_sequence = np.zeros((self.N_LOUDNESS, self.N_FFT_FRAMES))
		self.pitch_sequence = np.zeros((self.N_PITCH, self.N_FFT_FRAMES))
		self.onsets_sequence = []
		self.binaryRhythms_sequence = []

		# checking features received
		self.N_FEATURES = self.N_CHROMA + self.N_MELBANDS + self.N_SPECTRALSHAPE + self.N_LOUDNESS + self.N_ONSET + self.N_PITCH
		self.EXPECTED_NUM_FEATURES = self.N_FEATURES * (self.N_LOOPS + 1)
		self.featuresInCounter = 0

		# AUTONOMOUS LOOPER STATE VARIABLES
		self.bars_loop_persisted = np.zeros((self.N_LOOPS)).tolist()
		self.selected_loops_satisfaction_degrees = [0 for _ in range(self.N_LOOPS)]
		self.active_loops = [False for _ in range(self.N_LOOPS)]
		#previous_bars = [np.zeros(self.BEAT_SAMPLES * self.BEATS_PER_LOOP) for _ in range(self.N_BARS_STARTUP-1)]
		self.silence_threshold = -10000
		self.previous_descriptors = [[] for _ in range(self.N_BARS_STARTUP)]
		self.all_descriptors = []
		self.BARS_COUNT = 0
		
		# compute candidate segment divisions
		self.candidate_segments_divisions = []
		for n in range(self.MIN_BEATS_PER_LOOP, self.BEATS_PER_LOOP, self.MIN_BEATS_PER_LOOP):
			if self.BEATS_PER_LOOP % n == 0:
				self.candidate_segments_divisions.append(n)
		self.candidate_segments_divisions.append(self.BEATS_PER_LOOP)
		self.min_loop_division = self.candidate_segments_divisions[0]
		print(f'Candidate segment divisions: {self.candidate_segments_divisions}')		

		## OSC ADDRESSES
		# network parameters
		self.ip = ip # localhost
		self.port_snd = port_snd # send port to PD
		self.port_rcv = port_rcv # receive port from PD

		# LOAD PURE DATA LOOPER
		t = threading.Thread(target=self.launchPD,name='ALL_PD',args=(UBUNTU,))
		#t.daemon = True
		t.start()

		# OSC SERVER
		dispatcher = Dispatcher()
		dispatcher.map("/features/*", self.liveFeaturesIn_handler)
		dispatcher.set_default_handler(self.default_handler)

		# define client
		self.client = udp_client.SimpleUDPClient(self.ip, self.port_snd)

		# define server
		self.server = BlockingOSCUDPServer((self.ip, self.port_rcv), dispatcher)
		self.server.serve_forever()  # Blocks forever

	def launchPD(self, UBUNTU):
		if not UBUNTU:
			pd_looper_path = './02_ALL_PD/_main_v2.pd'
			#command = macos_pd_executable + f' -send "; N_LOOPS {self.N_LOOPS}; BPM {self.TEMPO}; fftsize {self.FFT_WINDOW}; hopsize {self.FFT_HOP_SIZE}; " ' + pd_looper_path
			command = macos_pd_executable + f' -send "; N_LOOPS {self.N_LOOPS}; BPM {self.TEMPO}; BEATS_PER_LOOP {self.BEATS_PER_LOOP}; PORT_SND {self.port_rcv}; PORT_RCV {self.port_snd}; " ' + pd_looper_path
			os.system(command)

	def default_handler(self, address, *args):
	    print(f"DEFAULT {address}: {len(args)}")

	def liveFeaturesIn_handler(self, address, *args):
		#print(f"{address}: {len(args)}")
		feature_name = address.split('/')[-1].split('-')[0]
		loop_num = int(address.split('/')[2])
		feature_component_num = int(address.split('/')[-1].split('-')[-1])

		# SIMPLE FEATURE RECEIVER
		if loop_num == 1000:
			if feature_name == 'chroma':
				self.chroma_sequence[feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'spectralshape':
				self.spectralshape_sequence[feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'melbands':
				self.melbands_sequence[feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'loudness':
				self.loudness_sequence[feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'pitch':
				self.pitch_sequence[feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'onsets':
				self.onsets = np.abs(np.array(args))
				self.binaryRhythms_sequence = self.getBinaryRhythm(self.onsets)
				'''
				interval_size = int(self.N_BAR_SAMPLES / self.RHYTHM_SUBDIVISIONS)
				binary_rhythm = []
				for i in range(0, self.N_BAR_SAMPLES, interval_size):
					# if there is a onset in the bar division 1, otherwise 0
					flag = 0
					flag_dynamic = 0
					for onset in self.onsets:
						if onset > i and onset <= i+interval_size:
							flag = 1
					binary_rhythm.append(flag)
				self.onsets_sequence = (self.onsets / self.FFT_HOP_SIZE).astype(int)
				self.binaryRhythms_sequence = binary_rhythm
				'''
				self.featuresInCounter += 1
		else:
			if feature_name == 'chroma':
				self.chroma_loops[loop_num, feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'spectralshape':
				self.spectralshape_loops[loop_num, feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'melbands':
				self.melbands_loops[loop_num, feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'loudness':
				self.loudness_loops[loop_num, feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'pitch':
				self.pitch_loops[loop_num, feature_component_num, :] = np.array(args)[:self.N_FFT_FRAMES]
				self.featuresInCounter += 1
			elif feature_name == 'onsets':
				self.onsets = np.abs(np.array(args))
				self.onsets_loops[loop_num] = (self.onsets / self.FFT_HOP_SIZE).astype(int)
				# binary rhythm representation
				interval_size = int(self.N_BAR_SAMPLES / self.RHYTHM_SUBDIVISIONS)
				binary_rhythm = []
				for i in range(0, self.N_BAR_SAMPLES, interval_size):
					# if there is a onset in the bar division 1, otherwise 0
					flag = 0
					flag_dynamic = 0
					for onset in self.onsets:
						if onset > i and onset <= i+interval_size:
							flag = 1
					binary_rhythm.append(flag)
				self.binaryRhythms_loops[loop_num] = binary_rhythm
				#print(binary_rhythm)
				self.featuresInCounter += 1

		# ACTION WHEN ALL FEATURES HAVE BEEN RECIEVED
		if self.featuresInCounter >= self.EXPECTED_NUM_FEATURES:
			self.featuresInCounter = 0
			print()
			print(f'SEGMENT {self.BARS_COUNT}')
			print('-'*50)

			# compute tonnetz from chroma
			self.tonnetz_sequence = librosa.feature.tonnetz(chroma=self.chroma_sequence, sr=self.sr)
			for i in range(self.N_LOOPS):
				self.tonnetz_loops[i,:,:] = librosa.feature.tonnetz(chroma=self.chroma_loops[i,:,:], sr=self.sr)

			# compute bar loudness
			bar_mean_loudness = self.loudness_sequence[0, :].mean() # mean loudness of bar
			if not any(self.active_loops):
				# INITIAL OPERATIONAL MODE
				# check that the bar is not completely silent
				if bar_mean_loudness > self.silence_threshold:
					bar_sequence_descriptors = self.getCurrentSequenceDescriptors()
					if self.BARS_COUNT > self.N_BARS_STARTUP:
						# compare descriptors of current bar with previous bar
						previous_metrics = []
						for k in range(len(self.previous_descriptors)):
							# compute comparison coefficients
							comparison_metrics = self.compareSequenceWithLoops(bar_sequence_descriptors, self.previous_descriptors[k], self.RHYTHM_SUBDIVISIONS)
							previous_metrics.append(comparison_metrics)
						for i in range(self.N_LOOPS):
							if not any(self.active_loops): # check that loops haven't been activated in the meantime
								# check if repetition rules are satisfied
								rules_satisfied, satisfaction_degree = self.evaluateStartupRepetitionCriteria(self.looping_rules[i], previous_metrics, comparison_metrics)
								print(f'Loop {i+1}')
								print(f'Rule satisfaction degree {satisfaction_degree:.3f}')
								#if all(rules_satisfied): 
								if satisfaction_degree >= self.STARTUP_SIMILARITY_THR: 
									print(f'Segment selected for loop {i+1}')
									self.client.send_message(f"/loopdecision/loop/{self.candidate_segments_divisions[-1]}", str(i))
									self.bars_loop_persisted[i] = 0
									self.active_loops[i] = True
					newdescriptors = []
					for descriptor in bar_sequence_descriptors:
						newdescriptors.append(descriptor.copy())
					self.previous_descriptors.append(newdescriptors)
					del self.previous_descriptors[0] # remove firts element of bar list (make circular buffer)
			
			else:

				# BASIC OPERATIONAL MODE
				all_loops_satisfaction_degrees = [0 for _ in range(self.N_LOOPS)]
				all_loops_rules_satisfied = [False for _ in range(self.N_LOOPS)]
				selected_candidate_nums = [0 for _ in range(self.N_LOOPS)]
				# COMPUTE COMPARISON METRICS
				for i in range(self.N_LOOPS):

					# get descriptors
					#bar_sequence_descriptors = self.getCurrentSequenceDescriptors()
					sumOfLoops_sequence_descriptors = self.getSumOfLoopsDescriptors(i)

					# compute candidate segments based on segment divisions
					candidate_segments_descriptors = []
					for n in self.candidate_segments_divisions:
						candidate_segment_descriptors = self.getCurrentSequenceDescriptorsWDivisions(n)
						candidate_segments_descriptors.append(candidate_segment_descriptors)

					# Check for multiple candidate segments
					candidates_satisfaction_degrees = []
					# compute rules for each candidate segment
					for segment_descriptors in candidate_segments_descriptors:
						# COMPUTE COMPARISON METRICS
						comparison_metrics = self.compareSequenceWithLoops(segment_descriptors, sumOfLoops_sequence_descriptors, self.RHYTHM_SUBDIVISIONS)
						# EVALUATE LOOPING RULES
						rules_satisfied, rules_satisfaction_degree = self.evaluateLoopingRules(self.looping_rules[i], comparison_metrics)
						cumulative_satisfaction_degree = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
						# set satisfaction degree to 0 if rules not satisfied
						if all(rules_satisfied):
							candidates_satisfaction_degrees.append(cumulative_satisfaction_degree)
						else:
							candidates_satisfaction_degrees.append(0)

					max_candidates_satisfaction_degree = np.argmax(np.array(candidates_satisfaction_degrees))
					all_loops_satisfaction_degrees[i] = candidates_satisfaction_degrees[max_candidates_satisfaction_degree]
					all_loops_rules_satisfied[i] = True if candidates_satisfaction_degrees[max_candidates_satisfaction_degree] != 0 else False
					selected_candidate_nums[i] = max_candidates_satisfaction_degree
					
					print(f'Loop track L_{i+1}')
					print(f'Most satisfactory candidate is segment {max_candidates_satisfaction_degree+1}')
					print(f'Rule satisfaction degree {all_loops_satisfaction_degrees[i]:.3f}')

					#comparison_metrics = self.compareSequenceWithLoops(bar_sequence_descriptors, sumOfLoops_sequence_descriptors, self.RHYTHM_SUBDIVISIONS)
					# EVALUATE LOOPING RULES
					#rules_satisfied, rules_satisfaction_degree = self.evaluateLoopingRules(self.looping_rules[i], comparison_metrics)
					#all_loops_satisfaction_degrees[i] = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
					#all_loops_rules_satisfied[i] = all(rules_satisfied)
					#print(f'Loop {i+1}')
					#print(f'Rule satisfaction degree {all_loops_satisfaction_degrees[i]:.3f}')
				
				# UPDATE LOOPS
				all_loops_satisfaction_degrees = [all_loops_satisfaction_degrees[i] if all_loops_rules_satisfied[i] else 0 for i in range(len(all_loops_satisfaction_degrees))]
				#all_loops_satisfaction_degrees = np.sort(np.array(all_loops_satisfaction_degrees)).tolist()
				#for i in range(self.N_LOOPS):
				loops_sorted_by_satisfaction_degree = np.argsort(np.array(all_loops_satisfaction_degrees)).tolist()
				for i in range(len(loops_sorted_by_satisfaction_degree)):				
					# CHECK IF LOOP SHOULD BE UPDATED
					if all_loops_rules_satisfied[i]:
						if self.bars_loop_persisted[i] >= self.MIN_LOOPS_REPETITION:
							if self.LOOP_CHANGE_RULE == "newer":
								print('')
								print('-'*50)
								print(f'Bar {self.BARS_COUNT} selected for loop {i+1}')
								print('-'*50)
								self.client.send_message(f"/loopdecision/loop/{str(self.candidate_segments_divisions[selected_candidate_nums[i]])}", str(i))
								self.bars_loop_persisted[i] = 0
								self.active_loops[i] = True
								self.selected_loops_satisfaction_degrees[i] = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
								break

							elif self.LOOP_CHANGE_RULE == "better":
								if sum(rules_satisfaction_degree)/len(rules_satisfaction_degree) >= self.selected_loops_satisfaction_degrees[i]:
									print('')
									print('-'*50)
									print(f'Bar {self.BARS_COUNT} selected for loop {i+1}')
									print('')
									self.client.send_message(f"/loopdecision/loop/{str(self.candidate_segments_divisions[selected_candidate_nums[i]])}", str(i))
									self.bars_loop_persisted[i] = 0
									self.active_loops[i] = True
									self.selected_loops_satisfaction_degrees[i] = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
									break
				

				# CHECK IF LOOP SHOULD BE DROPPED
				for i in range(self.N_LOOPS):
					if self.bars_loop_persisted[i] >= self.MAX_LOOPS_REPETITION:
						self.client.send_message("/loopdecision/drop", str(i))
						self.bars_loop_persisted[i] = 0
						self.selected_loops_satisfaction_degrees[i] = 0
						self.active_loops[i] = False
					else: 
						self.bars_loop_persisted[i] += 1
			
			self.BARS_COUNT += 1

	def evaluateLoopingRules(self, looping_rules, comparison_metrics):

		# EVALUATE LOOP CANDIDATE BASED ON RULES COMBINATION
		n_rule_components = len(looping_rules)
		rules_satisfied = []
		rules_satisfaction_degree = [] # between 0 and 1
		# iterate over rule components
		for rule in looping_rules:
			rule_satisfied = False
			rule_satisfaction_degree = 0
			threshold = rule["rule-threshold"]

			for i in range(len(self.RULE_NAMES)):
				if rule["rule-name"] == self.RULE_NAMES[i]:
					if rule["rule-type"] == "more":
						if comparison_metrics[i] >= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(comparison_metrics[i] - threshold)
					elif rule["rule-type"] == "less":
						if comparison_metrics[i] <= threshold:
							rule_satisfied = True
							rule_satisfaction_degree = abs(threshold - comparison_metrics[i])

			rules_satisfied.append(rule_satisfied)
			rules_satisfaction_degree.append(rule_satisfaction_degree)

		return rules_satisfied, rules_satisfaction_degree


	def evaluateStartupRepetitionCriteria(self, looping_rules, previous_metrics, comparison_metrics):

		satisfaction_degree = 0
		rules_satisfied = []
		for rule in looping_rules:
			for i in range(len(self.RULE_NAMES)):
				if rule["rule-name"] == self.RULE_NAMES[i]:
					# get descriptor values correpsonding to these metrics
					metrics_values = [metrics_bar[i] for metrics_bar in previous_metrics]
					satisfaction_degree += np.sum(metrics_values) / len(metrics_values)
					if all(value > self.STARTUP_SIMILARITY_THR for value in metrics_values):
						rules_satisfied.append(True)
					else: 
						rules_satisfied.append(False)

		satisfaction_degree /= len(looping_rules)
		return rules_satisfied, satisfaction_degree


	def getCurrentSequenceDescriptors(self):

		# process features
		# current sequence features
		onsets_seq = self.onsets_sequence
		binary_rhythm_seq = self.binaryRhythms_sequence
		CQT_seq = self.melbands_sequence
		CQT_mean = CQT_seq.mean(axis=1)
		CQT_center_of_mass_seq = sum([CQT_mean[i]*i for i in range(CQT_mean.shape[0])]) / sum(CQT_mean)
		CQT_var_seq = np.std([CQT_mean[i]*i for i in range(CQT_mean.shape[0])])
		chroma_seq = self.chroma_sequence[:,:]
		discretechroma_seq = np.array([chroma_seq[:,j] for j in onsets_seq])
		tonnetz_seq = self.tonnetz_sequence[:,:]
		discretetonnetz_seq = np.array([tonnetz_seq[:,j] for j in onsets_seq])
		loudness_seq = self.loudness_sequence[0,:]
		discreteloudness_seq = np.array([loudness_seq[j] for j in onsets_seq])
		#centroid_seq = self.spectralshape_sequence[0,:]
		centroid_seq = self.pitch_sequence[0,:]
		discretecentroid_seq = np.array([centroid_seq[j] for j in onsets_seq])
		flatness_seq = self.spectralshape_sequence[5,:]
		discreteflatness_seq = np.array([flatness_seq[j] for j in onsets_seq])

		bar_sequence_descriptors = [binary_rhythm_seq, np.array(onsets_seq), 
									CQT_seq, CQT_center_of_mass_seq, CQT_var_seq,
									chroma_seq, discretechroma_seq,
									loudness_seq.reshape(1,-1), discreteloudness_seq,
									centroid_seq.reshape(1,-1), discretecentroid_seq,
									flatness_seq.reshape(1,-1), discreteflatness_seq,
									tonnetz_seq, discretetonnetz_seq]
		
		return bar_sequence_descriptors


	def getCurrentSequenceDescriptorsWDivisions(self, n=1):

		segment_length_frames = int(self.frames_per_beat*n)+1

		# process features
		# current sequence features
		onsets_seq = self.onsets_sequence
		binary_rhythm_seq = self.binaryRhythms_sequence

		# CQT
		candidate_segment = np.zeros((self.N_MELBANDS, self.N_FFT_FRAMES))
		segment = self.melbands_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		CQT_seq = candidate_segment.copy()
		CQT_mean = CQT_seq.mean(axis=1)
		CQT_center_of_mass_seq = sum([CQT_mean[i]*i for i in range(CQT_mean.shape[0])]) / sum(CQT_mean)
		CQT_var_seq = np.std([CQT_mean[i]*i for i in range(CQT_mean.shape[0])])
		
		# chroma
		candidate_segment = np.zeros((self.N_CHROMA, self.N_FFT_FRAMES))
		segment = self.chroma_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		chroma_seq = candidate_segment.copy()
		discretechroma_seq = np.array([chroma_seq[:,j] for j in onsets_seq])

		# tonnetz
		candidate_segment = np.zeros((self.N_TONNETZ, self.N_FFT_FRAMES))
		segment = self.tonnetz_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		tonnetz_seq = candidate_segment.copy()
		discretetonnetz_seq = np.array([tonnetz_seq[:,j] for j in onsets_seq])
		
		# loudness
		candidate_segment = np.zeros((self.N_LOUDNESS, self.N_FFT_FRAMES))
		segment = self.loudness_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		loudness_seq = candidate_segment.copy()[0,:]
		discreteloudness_seq = np.array([loudness_seq[j] for j in onsets_seq])

		# centroid
		candidate_segment = np.zeros((self.N_SPECTRALSHAPE, self.N_FFT_FRAMES))
		segment = self.spectralshape_sequence[:,-segment_length_frames:]
		#num_repetitions = int(self.BEATS_PER_LOOP / n)
		for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))+1):
			candidate_segment[:,k*segment_length_frames:(k+1)*segment_length_frames] = np.array(segment)
		
		centroid_seq = candidate_segment.copy()[0,:]
		discretecentroid_seq = np.array([centroid_seq[j] for j in onsets_seq])

		flatness_seq = candidate_segment.copy()[5,:]
		discreteflatness_seq = np.array([flatness_seq[j] for j in onsets_seq])


		bar_sequence_descriptors = [binary_rhythm_seq, np.array(onsets_seq), 
									CQT_seq, CQT_center_of_mass_seq, CQT_var_seq,
									chroma_seq, discretechroma_seq,
									loudness_seq.reshape(1,-1), discreteloudness_seq,
									centroid_seq.reshape(1,-1), discretecentroid_seq,
									flatness_seq.reshape(1,-1), discreteflatness_seq,
									tonnetz_seq, discretetonnetz_seq]
		
		return bar_sequence_descriptors


	def getBinaryRhythm(self, onsets):

		interval_size = int(self.N_BAR_SAMPLES / self.RHYTHM_SUBDIVISIONS)
		binary_rhythm = []
		for i in range(0, self.N_BAR_SAMPLES, interval_size):
			# if there is a onset in the bar division 1, otherwise 0
			flag = 0
			flag_dynamic = 0
			for onset in onsets:
				if onset > i and onset <= i+interval_size:
					flag = 1
			binary_rhythm.append(flag)
		return binary_rhythm


	def getSumOfLoopsDescriptors(self, loopNumber):

		# sum of loops features
		onsets_sum = self.onsets_loops[loopNumber]
		binary_rhythm_sum = self.binaryRhythms_loops[loopNumber]
		CQT_sum = self.melbands_loops[loopNumber]
		CQT_mean = CQT_sum.mean(axis=1)
		CQT_center_of_mass_sum = sum([CQT_mean[i]*i for i in range(CQT_mean.shape[0])]) / sum(CQT_mean)
		CQT_var_sum = np.std([CQT_mean[i]*i for i in range(CQT_mean.shape[0])])
		chroma_sum = self.chroma_loops[loopNumber,:,:]
		discretechroma_sum = np.array([chroma_sum[:,j] for j in onsets_sum])
		tonnetz_sum = self.tonnetz_loops[loopNumber,:,:]
		discretetonnetz_sum = np.array([tonnetz_sum[:,j] for j in onsets_sum])
		loudness_sum = self.loudness_loops[loopNumber,0,:]
		discreteloudness_sum = np.array([loudness_sum[j] for j in onsets_sum])
		#centroid_sum = self.spectralshape_loops[loopNumber,0,:]
		centroid_sum = self.pitch_loops[loopNumber,0,:]
		discretecentroid_sum = np.array([centroid_sum[j] for j in onsets_sum])
		flatness_sum = self.spectralshape_loops[loopNumber,5,:]
		discreteflatness_sum = np.array([flatness_sum[j] for j in onsets_sum])

		sumOfLoops_sequence_descriptors = [binary_rhythm_sum, np.array(onsets_sum), 
											CQT_sum, CQT_center_of_mass_sum, CQT_var_sum,
											chroma_sum, discretechroma_sum,
											loudness_sum.reshape(1,-1), discreteloudness_sum,
											centroid_sum.reshape(1,-1), discretecentroid_sum,
											flatness_sum.reshape(1,-1), discreteflatness_sum,
											tonnetz_sum, discretetonnetz_sum]

		return sumOfLoops_sequence_descriptors


	def compareSequenceWithLoops(self, bar_sequence_descriptors, sumOfLoops_sequence_descriptors, rhythm_subdivisions):

		if self.verbose == 2:
			print('Binary rhythms:')
		binary_comparison_coefficient, rhythm_density_coefficient = self.compareBinaryRhythms(bar_sequence_descriptors[0], sumOfLoops_sequence_descriptors[0], rhythm_subdivisions)

		## SPECTRAL BANDWIDTH
		if self.verbose == 2:
			print('Spectral bandwidth:')
		spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient = self.compareSpectralBandwidth(bar_sequence_descriptors[2], bar_sequence_descriptors[3], bar_sequence_descriptors[4], 
																													sumOfLoops_sequence_descriptors[2], sumOfLoops_sequence_descriptors[3], sumOfLoops_sequence_descriptors[4])

		## CHROMA
		if self.verbose == 2:
			print('Chroma:')
		chroma_AE = self.computeTwodimensionalAE(bar_sequence_descriptors[5], sumOfLoops_sequence_descriptors[5])
		_, chroma_continuous_correlation = self.computeTwodimensionalContinuousCorrelation(bar_sequence_descriptors[5], sumOfLoops_sequence_descriptors[5])
		_, chroma_discrete_correlation = self.computeTwodimensionalDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[6], 
																					sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[6])

		## TONNETZ
		if self.verbose == 2:
			print('Tonnetz:')
		tonnetz_AE = self.computeTwodimensionalAE(bar_sequence_descriptors[13], sumOfLoops_sequence_descriptors[13])
		_, tonnetz_continuous_correlation = self.computeTwodimensionalContinuousCorrelation(bar_sequence_descriptors[13], sumOfLoops_sequence_descriptors[13])
		_, tonnetz_discrete_correlation = self.computeTwodimensionalDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[14], 
																					sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[14])

		## LOUDNESS
		if self.verbose == 2:
			print('Loudness:')
		loudness_MSE = self.computeSignalsMSE(bar_sequence_descriptors[7], sumOfLoops_sequence_descriptors[7])
		_, loudness_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[7], sumOfLoops_sequence_descriptors[7])
		_, loudness_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[8], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[8])

		## CENTROID
		if self.verbose == 2:
			print('Spectral centroid:')
		centroid_MSE = self.computeSignalsMSE(bar_sequence_descriptors[9], sumOfLoops_sequence_descriptors[9])
		_, centroid_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[9], sumOfLoops_sequence_descriptors[9])
		_, centroid_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[10], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[10])

		## FLATNESS
		if self.verbose == 2:
			print('Spectral flatness:')
		flatness_MSE = self.computeSignalsMSE(bar_sequence_descriptors[11], sumOfLoops_sequence_descriptors[11])
		_, flatness_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[11], sumOfLoops_sequence_descriptors[11])
		_, flatness_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[12], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[12])
		if self.verbose == 2:
			print()

		# these have to match the order in self.RULE_NAMES
		comparison_metrics = [chroma_AE, chroma_continuous_correlation/2+0.5, chroma_discrete_correlation/2+0.5,
							centroid_MSE, centroid_continuous_correlation/2+0.5, centroid_discrete_correlation/2+0.5,
							loudness_MSE, loudness_continuous_correlation/2+0.5, loudness_discrete_correlation/2+0.5,
							flatness_MSE, flatness_continuous_correlation/2+0.5, flatness_discrete_correlation/2+0.5,
							spectral_energy_difference_coefficient, spectral_energy_overlap_coefficient,
							binary_comparison_coefficient, rhythm_density_coefficient]

		return comparison_metrics


	# FUNCTIONS TO COMPARE FEATURES
	def compareBinaryRhythms(self, binary_rhythm1, binary_rhythm2, rhythm_subdivisions=16):
		if self.verbose == 2:
			print(np.array(binary_rhythm1))
			print(np.array(binary_rhythm2))
		binary_comparison = [1 if binary_rhythm1[i] == binary_rhythm2[i] else 0 for i in range(len(binary_rhythm1))]
		binary_comparison_coefficient = sum(binary_comparison) / rhythm_subdivisions 
		if self.verbose == 2:
			print(f'Binary Comparison coefficient: {binary_comparison_coefficient:.3f}')
		rhythm_density_coefficient = abs(np.array(binary_rhythm1).sum() - np.array(binary_rhythm2).sum()) / rhythm_subdivisions
		if self.verbose == 2:
			print(f"Rhythm Density Comparison coefficient: {rhythm_density_coefficient:.3f}")
		return binary_comparison_coefficient, rhythm_density_coefficient

	def compareSpectralBandwidth(self, CQT1, CQT1_center_of_mass, CQT1_var, CQT2, CQT2_center_of_mass, CQT2_var, plotflag=False):
		CQT1_mean = CQT1.mean(axis=1)
		CQT2_mean = CQT2.mean(axis=1)
		spectral_energy_overlap_index = max(0, 
		    min(CQT1_center_of_mass+CQT1_var, CQT2_center_of_mass+CQT2_var) - max(CQT1_center_of_mass-CQT1_var, CQT2_center_of_mass-CQT2_var))
		spectral_energy_overlap_coefficient = min(spectral_energy_overlap_index, min(CQT1_var*2, CQT2_var*2)) / min(CQT1_var*2, CQT2_var*2)
		if self.verbose == 2:
			print(f"Spectral energy overlap coefficient: {spectral_energy_overlap_coefficient:.3f}")
		spectral_energy_difference_coefficient = np.abs(CQT1_mean - CQT2_mean).mean()
		if self.verbose == 2:
			print(f"Spectral energy difference coefficient: {spectral_energy_difference_coefficient:.3f}")
		return spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient

	def computeSignalsMSE(self, signal1, signal2):
		minsignal = min(signal1.min(), signal2.min())
		maxsignal = max(signal1.max(), signal2.max())
		normalized_signal1 = (signal1 - minsignal) / (maxsignal - minsignal)
		normalized_signal2 = (signal2 - minsignal) / (maxsignal - minsignal)
		MSE = ((normalized_signal1 - normalized_signal2)**2).mean()
		if self.verbose == 2:
			print(f'MSE between the two signal is: {MSE:.3f}')
		return MSE

	def computeContinuousCorrelation(self, signal1, signal2):
		if (signal1.max() - signal1.min()) != 0 and (signal2.max() - signal2.min()) != 0:
			normalized_signal1 = (signal1 - signal1.min()) / (signal1.max() - signal1.min())
			normalized_signal2 = (signal2 - signal2.min()) / (signal2.max() - signal2.min())
			signal_time_correlation = normalized_signal1 * normalized_signal2
			time_correlation_coefficient = np.mean(signal_time_correlation.sum(axis=1) / signal_time_correlation.shape[1])
			pearson_correlation = ((signal1 - signal1.mean()) * (signal2 - signal2.mean()) / (signal1.std() * signal2.std())).mean()
			if np.isnan(time_correlation_coefficient):
				time_correlation_coefficient = 0
			if np.isnan(pearson_correlation):
				pearson_correlation = 0
			if self.verbose == 2:
				print(f"Continuous pearson correlation coefficient: {pearson_correlation:.3f}")
		else:
			time_correlation_coefficient = 0
			pearson_correlation = 0
			if self.verbose == 2:
				print(f"Continuous pearson correlation coefficient: {pearson_correlation:.3f}")
		return time_correlation_coefficient, pearson_correlation

	def computeDiscreteCorrelation(self, onsets1, values1, onsets2, values2, plotflag=False):
		if onsets1.shape[0] != 0 and onsets2.shape[0] != 0:
			values1 = np.append(values1, values1[-1])
			values2 = np.append(values2, values2[-1])
			# make arrays same size
			both_onsets = np.unique(np.sort(np.concatenate((onsets1, onsets2), axis=0)))
			# join arrays
			dtype = [('key', int), ('field', float)]
			x = np.array([(onsets1[i], values1[i]) for i in range(onsets1.shape[0])], dtype=dtype)
			y = np.array([(onsets2[i], values2[i]) for i in range(onsets2.shape[0])], dtype=dtype)
			join = np.lib.recfunctions.join_by('key', x, y, jointype='outer')
			join.fill_value = 1e10
			join = join.filled()
			X = [value[1] for value in join]
			for i in range(len(X)):
			    if X[i] != 1e10:
			        X[i] = X[i]
			    else:
			        X[i] = X[i-1]
			Y = [value[2] for value in join]
			for i in range(len(Y)):
			    if Y[i] != 1e10:
			        Y[i] = Y[i]
			    else:
			        Y[i] = Y[i-1]
			# join arrays
			new_values1 = np.array(X)
			new_values2 = np.array(Y)
			normalized_values1 = (new_values1 - new_values1.min()) / (new_values1.max() - new_values1.min())
			normalized_values2 = (new_values2 - new_values2.min()) / (new_values2.max() - new_values2.min())
			normalized_values_time_correlation = normalized_values1 * normalized_values2				
			discrete_time_correlation_coefficient = np.mean(normalized_values_time_correlation.sum() / normalized_values_time_correlation.shape[0])
			discrete_pearson_correlation = ((new_values1 - new_values1.mean()) * (new_values2 - new_values2.mean()) / (new_values1.std() * new_values2.std())).mean()
			if np.isnan(discrete_time_correlation_coefficient):
				discrete_time_correlation_coefficient = 0
			if np.isnan(discrete_pearson_correlation):
				discrete_pearson_correlation = 0
		else: 
			discrete_time_correlation_coefficient = 0
			discrete_pearson_correlation = 0
		if self.verbose == 2:
			print(f"Discrete pearson correlation coefficient: {discrete_pearson_correlation:.3f}")

		return discrete_time_correlation_coefficient, discrete_pearson_correlation

	def computeTwodimensionalMSE(self, signal1, signal2):
		minsignal = np.concatenate((signal1.min(axis=1).reshape(-1,1), signal2.min(axis=1).reshape(-1,1)), axis=1).min(axis=1).reshape(-1,1)
		maxsignal = np.concatenate((signal1.max(axis=1).reshape(-1,1), signal2.max(axis=1).reshape(-1,1)), axis=1).max(axis=1).reshape(-1,1)
		normalized_signal1 = (signal1 - minsignal) / (maxsignal - minsignal)
		normalized_signal2 = (signal2 - minsignal) / (maxsignal - minsignal)
		MSE = ((normalized_signal1 - normalized_signal2)**2).mean(axis=1).mean()
		if self.verbose == 2:
			print(f'MSE between the two signal is: {MSE:.3f}')
		return MSE

	def computeTwodimensionalAE(self, signal1, signal2, plotflag=False):
		similarity_coefficient = np.abs(np.mean(signal1.mean(axis=1) - signal2.mean(axis=1)))
		if self.verbose == 2:
			print(f"Absolute Error difference coefficient: {similarity_coefficient:.3f}")
		return similarity_coefficient

	def computeTwodimensionalContinuousCorrelation(self, signal1, signal2, plotflag=False):
		time_correlation = signal1 * signal2
		time_correlation_coefficient = np.mean(time_correlation.sum(axis=1) / time_correlation.shape[1])
		continuous_pearson_correlation = ((signal1 - signal1.mean(axis=1).reshape(-1,1)) * (signal2 - signal2.mean(axis=1).reshape(-1,1))).mean(axis=1) / (signal1.std(axis=1) * signal2.std(axis=1))
		if np.isnan(np.sum(time_correlation_coefficient)):
			time_correlation_coefficient = np.array([0])
		if np.isnan(np.sum(continuous_pearson_correlation)):
			continuous_pearson_correlation = np.array([0])
		if self.verbose == 2:
			print(f"Continuous pearson correlation coefficient: {continuous_pearson_correlation.mean():.3f}")
		return time_correlation_coefficient, continuous_pearson_correlation.mean()

	def computeTwodimensionalDiscreteCorrelation(self, onsets1, values1, onsets2, values2):
		if onsets1.shape[0] == 0 or onsets2.shape[0] == 0:
			discrete_time_correlation_coefficient = np.array([0])
			discrete_pearson_correlation = np.array([0])
		else:
			try:
				# this is due to a bug in np.lib.recfunctions.join_by() with a list in 'field' 
				values1 = np.concatenate((values1, values1[-1,:].reshape(1,-1)), axis=0)
				values2 = np.concatenate((values2, values2[-1,:].reshape(1,-1)), axis=0)
				# make arrays same size
				both_onsets = np.unique(np.sort(np.concatenate((onsets1, onsets2), axis=0)))
				dtype = [('key', int), ('field', list)]
				x = np.array([(int(onsets1[i]), values1[i,:].tolist()) for i in range(onsets1.shape[0])], dtype=dtype)
				y = np.array([(int(onsets2[i]), values2[i,:].tolist()) for i in range(onsets2.shape[0])], dtype=dtype)
				join = np.lib.recfunctions.join_by('key', x, y, jointype='outer')
				join.fill_value = 1e10
				join = join.filled()
				X = [value[1] for value in join]
				for i in range(len(X)):
				    if X[i] != 1e10:
				        X[i] = X[i]
				    else:
				        X[i] = X[i-1]
				Y = [value[2] for value in join]
				for i in range(len(Y)):
				    if Y[i] != 1e10:
				        Y[i] = Y[i]
				    else:
				        Y[i] = Y[i-1]
				new_values1 = np.array(X)
				new_values2 = np.array(Y)
				values_time_correlation = new_values1 * new_values2
				discrete_time_correlation_coefficient = np.mean(values_time_correlation.sum(axis=0) / values_time_correlation.shape[0])
				discrete_pearson_correlation = ((new_values1 - new_values1.mean(axis=1).reshape(-1,1)) * (new_values2 - new_values2.mean(axis=1).reshape(-1,1))).mean(axis=1) / (new_values1.std(axis=1) * new_values2.std(axis=1))
				if np.isnan(np.sum(discrete_time_correlation_coefficient)):
					discrete_time_correlation_coefficient = np.array([0])
				if np.isnan(np.sum(discrete_pearson_correlation)):
					discrete_pearson_correlation = np.array([0])
			except:
				if self.verbose == 2:
					print('COULD NOT COMPUTE CORRELATION')
				discrete_time_correlation_coefficient = np.array([0])
				discrete_pearson_correlation = np.array([0])
			if self.verbose == 2:
				print(f"Discrete pearson correlation coefficient: {discrete_pearson_correlation.mean():.3f}")
		return discrete_time_correlation_coefficient, discrete_pearson_correlation.mean()


if __name__ == '__main__': 

	looper = AutonomousLooperOnline()


