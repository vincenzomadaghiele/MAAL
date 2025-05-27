import librosa
import os
import json
import shutil
import glob
import imageio
import colorsys

import pandas as pd
import numpy as np
import numpy.lib.recfunctions
import matplotlib.pyplot as plt
import matplotlib.style as ms
import soundfile as sf
from natsort import natsorted  


class AutonomousLooperOffline():

	def __init__(self, 
				soundfile_filepath,
				config_filepath='./config.json',
				max_signal_size=10000000,
				seed=None,
				plotFlag=False,
				makeVideo=False, 
				verbose=0):

		print()
		print('Initializing Autonomous Looper offline')
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

		# DEFINE ALL NAMES OF LOOPING CRITERIA
		# the rules in the list should be in the same order as the vector that computes them		
		self.RULE_NAMES = ["Harmonic similarity", "Harmonic movement - C", "Harmonic movement - D",
							"Melodic similarity", "Melodic trajectory - C", "Melodic trajectory - D",
							"Dynamic similarity", "Dynamic changes - C", "Dynamic changes - D",
							"Timbral similarity", "Timbral evolution - C", "Timbral evolution - D",
							"Global spectral overlap", "Frequency range overlap",
							"Rhythmic similarity", "Rhythmic density",
							"Harmonic function similarity", "Harmonic function transitions - C", "Harmonic function transitions - D"]

		# CONFIGURE LOOP STATION WITH OPTIONS
		self.PLOT_FLAG = plotFlag
		self.verbose = verbose
		self.MAKE_VIDEO = makeVideo
		self.N_LOOPS = len(self.looping_rules)
		self.MAX_SIGNAL_SIZE = max_signal_size

		# LOAD AUDIO TRACK  
		self.soundfile_filepath = soundfile_filepath
		self.signal, self.sr = librosa.load(soundfile_filepath, sr=44100, mono=True)
		self.signal = self.signal[:self.MAX_SIGNAL_SIZE]

		print()
		print(f'Loading soundfile: {soundfile_filepath}')
		print('-'*50)
		print('num samples: ', self.signal.shape)
		print('time: {:2.3f} seconds '.format(librosa.samples_to_time(self.signal.shape[0], sr=self.sr)))
		print()
		
		if self.PLOT_FLAG:
			plt.figure(figsize=(10, 3))
			librosa.display.waveshow(y=self.signal, sr=self.sr, color="blue")
			plt.title(soundfile_filepath)
			plt.show()

		# TEMPO
		tempo_bps = self.TEMPO / 60 # beats per second
		beat_seconds = 1 / tempo_bps # duration of one beat [seconds]
		self.BEAT_SAMPLES = int(beat_seconds * self.sr) # number of samples of one beat
		self.NUM_BEATS = int(self.signal.shape[0] / self.BEAT_SAMPLES) # number of beats in the track

		# make loop envelope
		self.crown_window = np.ones(self.BEAT_SAMPLES * self.BEATS_PER_LOOP)
		fade_perc = 0.01 # percentage of fade in/out
		fade_num_samples = int(fade_perc * self.BEAT_SAMPLES * self.BEATS_PER_LOOP)
		fade_in = np.linspace(0,1,fade_num_samples)
		fade_out = np.linspace(1,0,fade_num_samples)
		self.crown_window[0:fade_num_samples] = fade_in
		self.crown_window[self.crown_window.shape[0]-fade_num_samples:] = fade_out

		# compute candidate segment divisions
		self.candidate_segments_divisions = []
		for n in range(self.MIN_BEATS_PER_LOOP, self.BEATS_PER_LOOP, self.MIN_BEATS_PER_LOOP):
			if self.BEATS_PER_LOOP % n == 0:
				self.candidate_segments_divisions.append(n)
		self.candidate_segments_divisions.append(self.BEATS_PER_LOOP)
		print(f'Candidate segment divisions: {self.candidate_segments_divisions}')

		# subdivide into 
		self.min_loop_division = self.candidate_segments_divisions[0]
		self.candidate_segments_divisions = [int(k/self.min_loop_division) for k in self.candidate_segments_divisions]
		self.signal_subdivided_samples = [(i * self.BEAT_SAMPLES * self.min_loop_division) for i in range(int(self.NUM_BEATS / self.min_loop_division))] # samples at which each looped bar starts

	def computeLooperTrack(self, output_dir_path):

		# SET UTILS FOR ITERATION
		# global variables
		decisions_log = []
		loops_bars = [[] for _ in range(self.N_LOOPS)]
		loops_candidate_num = [[] for _ in range(self.N_LOOPS)]
		loops_audiotracks = np.zeros((self.N_LOOPS, self.signal.shape[0]))
		
		# state variables
		loops = [np.zeros(self.BEAT_SAMPLES * self.BEATS_PER_LOOP) for _ in range(self.N_LOOPS)] # currently running loops
		bars_loop_persisted = np.zeros((self.N_LOOPS)).tolist()
		selected_loops_satisfaction_degrees = [0 for _ in range(self.N_LOOPS)]
		active_loops = [False for _ in range(self.N_LOOPS)]
		
		# for intial state
		previous_bars = [np.zeros(self.BEAT_SAMPLES * self.BEATS_PER_LOOP) for _ in range(self.N_BARS_STARTUP-1)]
		silence_threshold = 0.0003
		user_set_bar_count = 0

		for bar_num in range(self.BEATS_PER_LOOP, len(self.signal_subdivided_samples)):

			print('')
			print(f'Segment {bar_num}')
			print('-' * 40)

			# update log
			decisions_bar = {}
			decisions_bar['subdivision_index (m)'] = bar_num
			decisions_bar['decisions'] = []

			candidate_segments = []
			for n in self.candidate_segments_divisions:

				#print(n)
				#print(int(self.signal_subdivided_samples[bar_num]) - int(self.signal_subdivided_samples[bar_num-n]))

				segment = self.signal[int(self.signal_subdivided_samples[bar_num - n]):int(self.signal_subdivided_samples[bar_num])]
				segment_mean_loudness = librosa.feature.rms(y=segment)[0].mean() # mean loudness of bar
				segment_length_samples = int(self.signal_subdivided_samples[bar_num]) - int(self.signal_subdivided_samples[bar_num-n])
				
				if segment_mean_loudness > silence_threshold:
					candidate_segment = np.zeros(self.BEAT_SAMPLES * self.BEATS_PER_LOOP)
					num_repetitions = int(self.BEATS_PER_LOOP / n)
					for k in range(int(self.BEATS_PER_LOOP / (n * self.min_loop_division))):
						candidate_segment[k*segment_length_samples:(k+1)*segment_length_samples] = segment
					candidate_segments.append(candidate_segment)
			
			bar = candidate_segments[-1].copy()
			bar_mean_loudness = librosa.feature.rms(y=bar)[0].mean() # mean loudness of bar

			if not any(active_loops):
				# STARTUP CONDITIONS
				if self.STARTUP_MODE == "repetition":

					updated = False
					# check that the bar is not completely silent
					if bar_mean_loudness > silence_threshold:
						# compare descriptors of current bar with previous bar
						previous_metrics = []
						for previousbar in previous_bars:
							# compute comparison coefficients
							comparison_metrics = self.compareSequenceWithLoops(bar, [previousbar], self.sr, self.RHYTHM_SUBDIVISIONS)
							previous_metrics.append(comparison_metrics)
						
						# Compute rules satisfaction degrees for each track
						all_loops_satisfaction_degrees = [0 for _ in range(self.N_LOOPS)]
						for i in range(len(loops)):
							rules_satisfied, satisfaction_degree = self.evaluateStartupRepetitionCriteria(self.looping_rules[i], previous_metrics, comparison_metrics)
							print(f'Loop {i+1}')
							print(f'Rule satisfaction degree {satisfaction_degree:.3f}')
							all_loops_satisfaction_degrees[i] = satisfaction_degree

						# Start checking rules from the largest satisfaction degree
						loops_sorted_by_satisfaction_degree = np.argsort(np.array(all_loops_satisfaction_degrees)).tolist()
						for i in range(len(loops_sorted_by_satisfaction_degree)):
							if not any(active_loops): # check that loops haven't been activated in the meantime
								if all_loops_satisfaction_degrees[i] >= self.STARTUP_SIMILARITY_THR: 
									print('')
									print(f'Decision I_{i+1} ---> Segment selected for loop {i+1}')
									loops[i] = self.crown_window * bar
									loops_bars[i].append(bar_num)
									loops_candidate_num[i].append(0) # index of candidate in candidates array
									bars_loop_persisted[i] = 0
									active_loops[i] = True
									selected_loops_satisfaction_degrees[i] = all_loops_satisfaction_degrees[i]
									updated = True

									# update log
									decisions_element = {}
									decisions_element['decision_type'] = 'I'
									decisions_element['loop_track (i)'] = i
									decisions_element['num_beats (T_l)'] = self.candidate_segments_divisions[-1]
									decisions_element['satisfaction_degree'] = satisfaction_degree
									decisions_bar['decisions'].append(decisions_element)
								
								# UPDATE LOOPER AUDIOTRACKS
								if bar_num+1 < len(self.signal_subdivided_samples) - (self.BEATS_PER_LOOP / self.min_loop_division):
									loops_audiotracks[i,int(self.signal_subdivided_samples[bar_num+1]):int(self.signal_subdivided_samples[bar_num+1])+(self.BEAT_SAMPLES * self.BEATS_PER_LOOP)] = loops[i]

						previous_bars.append(bar)
						del previous_bars[0] # remove firts element of bar list (make circular buffer)
					
					if not updated:
						print('')
						print(f'Decision R ---> No updates')

						# update log
						decisions_element = {}
						decisions_element['decision_type'] = 'R'
						decisions_element['loop_track (i)'] = None
						decisions_element['num_beats (T_l)'] = None
						decisions_element['satisfaction_degree'] = None
						decisions_bar['decisions'].append(decisions_element)
				
				elif self.STARTUP_MODE == "user-set":
					
					updated = False
					user_set_bar_count += 1
					# START AFTER USER-SET NUMBER OF LOOP BEATS
					if user_set_bar_count == self.STARTUP_LOOP_BAR_N:
						# place user-set loop in bar
						loops[0] = self.crown_window * bar
						loops_bars[0].append(bar_num)
						loops_candidate_num[0].append(0) # index of candidate in candidates array
						bars_loop_persisted[0] = 0
						active_loops[0] = True
						user_set_bar_count = 0
						updated = True
						print('')
						print(f'Decision I_{i+1} ---> Segment selected for loop {i+1}')

						# update log
						decisions_element = {}
						decisions_element['decision_type'] = 'I'
						decisions_element['loop_track (i)'] = 0
						decisions_element['num_beats (T_l)'] = self.candidate_segments_divisions[-1]
						decisions_element['satisfaction_degree'] = None
						decisions_bar['decisions'].append(decisions_element)

						# UPDATE LOOPER AUDIOTRACKS
						if bar_num+1 < len(self.signal_subdivided_samples) - (self.BEATS_PER_LOOP / self.min_loop_division):
							loops_audiotracks[i,int(self.signal_subdivided_samples[bar_num+1]):int(self.signal_subdivided_samples[bar_num+1])+(self.BEAT_SAMPLES * self.BEATS_PER_LOOP)] = loops[i]

					if not updated:
						print('')
						print(f'Decision R ---> No updates')

						# update log
						decisions_element = {}
						decisions_element['decision_type'] = 'R'
						decisions_element['loop_track (i)'] = None
						decisions_element['num_beats (T_l)'] = None
						decisions_element['satisfaction_degree'] = None
						decisions_bar['decisions'].append(decisions_element)

			else:
				
				# CHECK RULES AND COMPUTE LOOP SATISFACTION DEGREES
				updated = False
				candidate_segments_for_loops = [np.zeros(self.BEAT_SAMPLES * self.BEATS_PER_LOOP) for _ in range(self.N_LOOPS)]
				all_loops_satisfaction_degrees = [0 for _ in range(self.N_LOOPS)]
				selected_candidate_nums = [0 for _ in range(self.N_LOOPS)]
				all_loops_rules_satisfied = [False for _ in range(self.N_LOOPS)]
				for i in range(len(loops)):
					loops_without_this = loops.copy()
					del loops_without_this[i]

					# Check for multiple candidate segments
					candidates_satisfaction_degrees = []
					# compute rules for each candidate segment
					for segment in candidate_segments:

						# COMPUTE COMPARISON METRICS
						comparison_metrics = self.compareSequenceWithLoops(segment, loops_without_this, self.sr, self.RHYTHM_SUBDIVISIONS)
						# EVALUATE LOOPING RULES
						rules_satisfied, rules_satisfaction_degree = self.evaluateLoopingRules(self.looping_rules[i], comparison_metrics)
						cumulative_satisfaction_degree = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
						
						# set satisfaction degree to 0 if rules not satisfied
						if all(rules_satisfied):
							candidates_satisfaction_degrees.append(cumulative_satisfaction_degree)
						else:
							candidates_satisfaction_degrees.append(0)

					max_candidates_satisfaction_degree = np.argmax(np.array(candidates_satisfaction_degrees))
					candidate_segments_for_loops[i] = candidate_segments[max_candidates_satisfaction_degree]
					all_loops_satisfaction_degrees[i] = candidates_satisfaction_degrees[max_candidates_satisfaction_degree]
					all_loops_rules_satisfied[i] = True if candidates_satisfaction_degrees[max_candidates_satisfaction_degree] != 0 else False
					selected_candidate_nums[i] = max_candidates_satisfaction_degree

					print(f'Loop track L_{i+1}')
					print(f'Most satisfactory candidate is segment {max_candidates_satisfaction_degree+1}')
					print(f'Rule satisfaction degree {all_loops_satisfaction_degrees[i]:.3f}')

				# CHECK LOOP UPDATES
				all_loops_satisfaction_degrees = [all_loops_satisfaction_degrees[i] if all_loops_rules_satisfied[i] else 0 for i in range(len(all_loops_satisfaction_degrees))]
				loops_sorted_by_satisfaction_degree = np.argsort(np.array(all_loops_satisfaction_degrees)).tolist()
				for i in range(len(loops_sorted_by_satisfaction_degree)):
					# CHECK IF LOOP SHOULD BE UPDATED
					if all_loops_rules_satisfied[i]:
						if bars_loop_persisted[i] >= self.MIN_LOOPS_REPETITION:
							if self.LOOP_CHANGE_RULE == "newer":
								print('')
								print(f'Decision A_{i+1} ---> Segment selected for loop {i+1}')
								loops[i] = self.crown_window * candidate_segments_for_loops[i]
								loops_bars[i].append(bar_num)
								loops_candidate_num[i].append(selected_candidate_nums[i]) # index of candidate in candidates array
								bars_loop_persisted[i] = 0
								active_loops[i] = True
								selected_loops_satisfaction_degrees[i] = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
								updated = True

								# update log
								decisions_element = {}
								decisions_element['decision_type'] = 'A'
								decisions_element['loop_track (i)'] = i
								decisions_element['num_beats (T_l)'] = self.candidate_segments_divisions[selected_candidate_nums[i]]
								decisions_element['satisfaction_degree'] = selected_loops_satisfaction_degrees[i]
								decisions_bar['decisions'].append(decisions_element)

								break
							elif self.LOOP_CHANGE_RULE == "better":
								if sum(rules_satisfaction_degree)/len(rules_satisfaction_degree) >= selected_loops_satisfaction_degrees[i]:
									print('')
									print(f'Decision A_{i+1} ---> Segment selected for loop {i+1}')
									loops[i] = self.crown_window * candidate_segments_for_loops[i]
									loops_bars[i].append(bar_num)
									loops_candidate_num[i].append(selected_candidate_nums[i]) # index of candidate in candidates array
									bars_loop_persisted[i] = 0
									active_loops[i] = True
									selected_loops_satisfaction_degrees[i] = sum(rules_satisfaction_degree)/len(rules_satisfaction_degree)
									updated = True

									# update log
									decisions_element = {}
									decisions_element['decision_type'] = 'A'
									decisions_element['loop_track (i)'] = i
									decisions_element['num_beats (T_l)'] = self.candidate_segments_divisions[selected_candidate_nums[i]]
									decisions_element['satisfaction_degree'] = selected_loops_satisfaction_degrees[i]
									decisions_bar['decisions'].append(decisions_element)

									break

				# CHECK IF LOOP SHOULD BE DROPPED
				for i in range(len(loops)):
					if bars_loop_persisted[i] >= self.MAX_LOOPS_REPETITION:
						print('')
						print(f'Decision Z_{i+1} ---> Clearing loop {i+1} audio buffer')
						loops[i] = np.zeros(self.BEAT_SAMPLES * self.BEATS_PER_LOOP)
						bars_loop_persisted[i] = 0
						selected_loops_satisfaction_degrees[i] = 0
						active_loops[i] = False
						updated = True
						# save dropped to dict

						# update log
						decisions_element = {}
						decisions_element['decision_type'] = 'Z'
						decisions_element['loop_track (i)'] = i
						decisions_element['num_beats (T_l)'] = None
						decisions_element['satisfaction_degree'] = None
						decisions_bar['decisions'].append(decisions_element)

					else: 
						bars_loop_persisted[i] += 1
				if not updated:
					print('')
					print(f'Decision R ---> No updates')

					# update log
					decisions_element = {}
					decisions_element['decision_type'] = 'R'
					decisions_element['loop_track (i)'] = None
					decisions_element['num_beats (T_l)'] = None
					decisions_element['satisfaction_degree'] = None
					decisions_bar['decisions'].append(decisions_element)


				# UPDATE LOOPER AUDIOTRACKS
				for i in range(len(loops)):
					if bar_num+1 < len(self.signal_subdivided_samples) - (self.BEATS_PER_LOOP / self.min_loop_division):
						loops_audiotracks[i,int(self.signal_subdivided_samples[bar_num+1]):int(self.signal_subdivided_samples[bar_num+1])+(self.BEAT_SAMPLES * self.BEATS_PER_LOOP)] = loops[i]

			decisions_log.append(decisions_bar)


		# SAVE TO DISK
		filename = self.soundfile_filepath.split('/')[-1].split('.')[0]
		output_dir = f'{output_dir_path}/{filename}'
		if os.path.isdir(output_dir):
			shutil.rmtree(output_dir)
		os.mkdir(output_dir)


		with open(f'{output_dir}/decisions_log.json', 'w', encoding='utf-8') as f:
			json.dump(decisions_log, f, ensure_ascii=False, indent=4)

		# SAVE SOUND FILES TO DISK
		all_loops = loops_audiotracks.sum(axis=0) #/ self.N_LOOPS
		signals = []
		signals.append(self.signal)
		signals.append(all_loops)
		signals = np.stack(signals)
		print()
		print('Saving files...')
		sf.write(f'{output_dir}/signal_w_loops.wav', signals.T, self.sr, subtype='PCM_24')
		sf.write(f'{output_dir}/signal.wav', self.signal, self.sr, subtype='PCM_24')
		sf.write(f'{output_dir}/all_loops.wav', all_loops, self.sr, subtype='PCM_24')
		for n in range(self.N_LOOPS):
			sf.write(f'{output_dir}/loop{n}_audiotrack.wav', loops_audiotracks[n], self.sr, subtype='PCM_24')
		with open(f'{output_dir}/config.json', 'w', encoding='utf-8') as f:
			json.dump(self.config, f, ensure_ascii=False, indent=4)

 
		# COMPUTE SUMMARY FIGURE
		fig, ax = plt.subplots(self.N_LOOPS+1, figsize=(12,self.N_LOOPS*2-2), gridspec_kw={'height_ratios': np.ones((self.N_LOOPS)).tolist().insert(0,3)})
		fig.subplots_adjust(top=0.8)

		colors = [colorsys.hsv_to_rgb(np.random.random(), 0.8, 0.9) for n in range(self.N_LOOPS)]
		vertical_line_length = np.max(self.signal)+ 0.1
		librosa.display.waveshow(self.signal, sr=self.sr, ax=ax[0], label='original signal', alpha=0.4)
		# plot loops
		for n in range(self.N_LOOPS):
			librosa.display.waveshow(loops_audiotracks[n], sr=self.sr, ax=ax[n+1], label=f'loop {n+1}', alpha=0.7, color=colors[n])
			loop_in_sig = np.zeros_like(self.signal)
			for j, bar_num in enumerate(loops_bars[n]):
				#segment_subdivision = (self.BEAT_SAMPLES * self.BEATS_PER_LOOP) / ((loops_candidate_num[n][j]+1) * self.MIN_BEATS_PER_LOOP)
				num_samples_selected_segment = self.candidate_segments_divisions[loops_candidate_num[n][j]] * self.min_loop_division * self.BEAT_SAMPLES
				loop_in_sig[int(self.signal_subdivided_samples[bar_num-1]):int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment] = self.signal[int(self.signal_subdivided_samples[bar_num-1]):int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment]
				ax[0].vlines(librosa.samples_to_time(int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment, sr=self.sr), -1*vertical_line_length, vertical_line_length, color=colors[n], alpha=0.9, linestyle='--', lw=0.8)
				ax[n+1].vlines(librosa.samples_to_time(int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment, sr=self.sr), -1*vertical_line_length, vertical_line_length, color='black', alpha=0.8, linestyle='--', lw=0.8)
			ax[0].plot(librosa.samples_to_time(np.arange(0, self.signal.shape[0]), sr=self.sr), loop_in_sig, label=f'loop {n+1}', alpha=0.6, color=colors[n])
			ax[n+1].xaxis.set_visible(False)
			ax[n+1].set_ylabel(f"$L_{n+1}$", rotation=0, ha='right', fontsize=13)
			ax[n+1].spines['right'].set_visible(False)
			ax[n+1].spines['top'].set_visible(False)
			ax[n+1].spines['bottom'].set_visible(False)
			if n != self.N_LOOPS:
				ax[n].set_xticklabels([])
			ax[n+1].set_yticklabels([])

		ax[n+1].xaxis.set_visible(True) # set last axis visible
		ax[n+1].set_xlabel("Time $[mm:ss]$")
		ax[n+1].spines['bottom'].set_visible(True)

		ax[0].spines['right'].set_visible(False)
		ax[0].spines['top'].set_visible(False)
		ax[0].spines['bottom'].set_visible(False)
		ax[0].set_xticklabels([])
		ax[0].set_yticklabels([])

		ax[0].set_ylabel("$x(t)$", rotation=0, ha='right', fontsize=13)
		ax[0].xaxis.set_visible(False)
		fig.suptitle(f'AUTONOMOUS LIVE LOOPER OUTPUT ON TRACK \n {self.soundfile_filepath.split('/')[-1]}', size=16, y=0.9)
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.savefig(f'{output_dir}/loops_figure.png')
		if self.PLOT_FLAG:
			plt.show()


		if self.MAKE_VIDEO:

			print('Generating cursor animation...')
			tempdir = f'{output_dir}/temp'
			if os.path.isdir(tempdir):
			    for f in os.listdir(tempdir):
			        os.remove(os.path.join(tempdir, f))
			    os.rmdir(tempdir)
			os.makedirs(tempdir)

			STEP = 60
			for step_count in range(0, int(self.signal.shape[0]), int(self.signal.shape[0]/STEP)):
				
				# COMPUTE SUMMARY FIGURE
				fig, ax = plt.subplots(self.N_LOOPS+1, figsize=(12,self.N_LOOPS*2-2), gridspec_kw={'height_ratios': np.ones((self.N_LOOPS)).tolist().insert(0,3)})
				fig.subplots_adjust(top=0.8)

				vertical_line_length = np.max(self.signal)+ 0.1
				librosa.display.waveshow(self.signal, sr=self.sr, ax=ax[0], label='original signal', alpha=0.4)
				# plot loops
				for n in range(self.N_LOOPS):
					librosa.display.waveshow(loops_audiotracks[n], sr=self.sr, ax=ax[n+1], label=f'loop {n+1}', alpha=0.7, color=colors[n])
					loop_in_sig = np.zeros_like(self.signal)
					for bar_num in loops_bars[n]:
						num_samples_selected_segment = self.candidate_segments_divisions[loops_candidate_num[n][j]] * self.min_loop_division * self.BEAT_SAMPLES
						loop_in_sig[int(self.signal_subdivided_samples[bar_num-1]):int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment] = self.signal[int(self.signal_subdivided_samples[bar_num-1]):int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment]
						ax[0].vlines(librosa.samples_to_time(int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment, sr=self.sr), -1*vertical_line_length, vertical_line_length, color=colors[n], alpha=0.9, linestyle='--', lw=0.8)
						ax[n+1].vlines(librosa.samples_to_time(int(self.signal_subdivided_samples[bar_num-1])+num_samples_selected_segment, sr=self.sr), -1*vertical_line_length, vertical_line_length, color='black', alpha=0.8, linestyle='--', lw=0.8)
					ax[0].plot(librosa.samples_to_time(np.arange(0, self.signal.shape[0]), sr=self.sr), loop_in_sig, label=f'loop {n+1}', alpha=0.6, color=colors[n])
					ax[n+1].xaxis.set_visible(False)
					ax[n+1].set_ylabel(f"$L_{n+1}$", rotation=0, ha='right', fontsize=13)
					ax[n+1].spines['right'].set_visible(False)
					ax[n+1].spines['top'].set_visible(False)
					ax[n+1].spines['bottom'].set_visible(False)
					if n != self.N_LOOPS:
						ax[n].set_xticklabels([])
					ax[n+1].set_yticklabels([])
					ax[n+1].vlines(librosa.samples_to_time(step_count, sr=self.sr), -1*vertical_line_length, vertical_line_length, color='r', alpha=1, lw=3)


				ax[n+1].xaxis.set_visible(True) # set last axis visible
				ax[n+1].set_xlabel("Time $[mm:ss]$")
				ax[n+1].spines['bottom'].set_visible(True)
				ax[0].vlines(librosa.samples_to_time(step_count, sr=self.sr), -1*vertical_line_length, vertical_line_length, color='r', alpha=1, lw=3)

				ax[0].spines['right'].set_visible(False)
				ax[0].spines['top'].set_visible(False)
				ax[0].spines['bottom'].set_visible(False)
				ax[0].set_xticklabels([])
				ax[0].set_yticklabels([])

				ax[0].set_ylabel("$x(t)$", rotation=0, ha='right', fontsize=13)
				ax[0].xaxis.set_visible(False)
				plt.subplots_adjust(wspace=0, hspace=0)
				fig.suptitle(f'AUTONOMOUS LIVE LOOPER OUTPUT ON TRACK \n {self.soundfile_filepath.split('/')[-1]}', size=16, y=0.9)
				plt.savefig(tempdir+'/image'+str(step_count)+'.png')
				plt.close()

			gif_file = f'{output_dir}/video.gif'
			with imageio.get_writer(gif_file, mode='I') as writer:
				filenames = glob.glob(tempdir+'/image*'+'.png')
				filenames = natsorted(filenames)
				for filename in filenames:
					image = imageio.v3.imread(filename)
					writer.append_data(image)
			shutil.rmtree(tempdir)


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


	def compareSequenceWithLoops(self, bar, loops, sr, rhythm_subdivisions):
		
		bar_sequence_descriptors = self.computeSequenceDescriptors(bar, sr, rhythm_subdivisions)
		sum_of_loops = np.zeros_like(bar)
		for loop in loops:
			sum_of_loops += loop
		sumOfLoops_sequence_descriptors = self.computeSequenceDescriptors(sum_of_loops, sr, rhythm_subdivisions)
	
		if self.verbose >= 1:
			print('Binary rhythms:')
		binary_comparison_coefficient, rhythm_density_coefficient = self.compareBinaryRhythms(bar_sequence_descriptors[0], sumOfLoops_sequence_descriptors[0], rhythm_subdivisions)

		## SPECTRAL BANDWIDTH
		if self.verbose >= 1:
			print('Spectral bandwidth:')
		spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient = self.compareSpectralBandwidth(bar_sequence_descriptors[2], bar_sequence_descriptors[3], bar_sequence_descriptors[4], 
																													sumOfLoops_sequence_descriptors[2], sumOfLoops_sequence_descriptors[3], sumOfLoops_sequence_descriptors[4])

		## CHROMA
		if self.verbose >= 1:
			print('Chroma:')
		chroma_AE = self.computeTwodimensionalAE(bar_sequence_descriptors[5], sumOfLoops_sequence_descriptors[5])
		_, chroma_continuous_correlation = self.computeTwodimensionalContinuousCorrelation(bar_sequence_descriptors[5], sumOfLoops_sequence_descriptors[5])
		_, chroma_discrete_correlation = self.computeTwodimensionalDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[6], 
																					sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[6])

		## TONNETZ
		if self.verbose >= 1:
			print('Chroma:')
		tonnetz_AE = self.computeTwodimensionalAE(bar_sequence_descriptors[13], sumOfLoops_sequence_descriptors[13])
		_, tonnetz_continuous_correlation = self.computeTwodimensionalContinuousCorrelation(bar_sequence_descriptors[13], sumOfLoops_sequence_descriptors[13])
		_, tonnetz_discrete_correlation = self.computeTwodimensionalDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[14], 
																					sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[14])

		## LOUDNESS
		if self.verbose >= 1:
			print('Loudness:')
		loudness_MSE = self.computeSignalsMSE(bar_sequence_descriptors[7], sumOfLoops_sequence_descriptors[7])
		_, loudness_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[7], sumOfLoops_sequence_descriptors[7])
		_, loudness_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[8], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[8])

		## PITCH
		if self.verbose >= 1:
			print('Pitch:')
		centroid_MSE = self.computeSignalsMSE(bar_sequence_descriptors[9], sumOfLoops_sequence_descriptors[9])
		_, centroid_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[9], sumOfLoops_sequence_descriptors[9])
		_, centroid_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[10], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[10])

		## FLATNESS
		if self.verbose >= 1:
			print('Spectral flatness:')
		flatness_MSE = self.computeSignalsMSE(bar_sequence_descriptors[11], sumOfLoops_sequence_descriptors[11])
		_, flatness_continuous_correlation = self.computeContinuousCorrelation(bar_sequence_descriptors[11], sumOfLoops_sequence_descriptors[11])
		_, flatness_discrete_correlation = self.computeDiscreteCorrelation(bar_sequence_descriptors[1], bar_sequence_descriptors[12], 
																	sumOfLoops_sequence_descriptors[1], sumOfLoops_sequence_descriptors[12])
		if self.verbose >= 1:
			print()

		# these have to match the order in self.RULE_NAMES
		comparison_metrics = [chroma_AE, chroma_continuous_correlation/2+0.5, chroma_discrete_correlation/2+0.5,
							centroid_MSE, centroid_continuous_correlation/2+0.5, centroid_discrete_correlation/2+0.5,
							loudness_MSE, loudness_continuous_correlation/2+0.5, loudness_discrete_correlation/2+0.5,
							flatness_MSE, flatness_continuous_correlation/2+0.5, flatness_discrete_correlation/2+0.5,
							spectral_energy_difference_coefficient, spectral_energy_overlap_coefficient,
							binary_comparison_coefficient, rhythm_density_coefficient,
							tonnetz_AE, tonnetz_continuous_correlation, tonnetz_discrete_correlation]

		return comparison_metrics


	def computeSequenceDescriptors(self, bar, sr, rhythm_subdivisions):

		# compute descriptors for a sequence
		binary_rhythm_bar, onsets, _, _ = self.computeBinaryRhythm(bar, sr, rhythm_subdivisions=rhythm_subdivisions, plotflag=self.PLOT_FLAG)
		CQT_bar, CQT_center_of_mass_bar, CQT_var_bar = self.computeCQT(bar, sr, plotflag=self.PLOT_FLAG)
		chroma_bar = self.computeChroma(bar, sr, plotflag=self.PLOT_FLAG)
		discretechroma_bar = self.computeDiscreteChroma(chroma_bar, onsets, sr, plotflag=self.PLOT_FLAG)
		tonnetz_bar = self.computeTonnetz(bar, sr, plotflag=self.PLOT_FLAG)
		discretetonnetz_bar = self.computeDiscreteChroma(tonnetz_bar, onsets, sr, plotflag=self.PLOT_FLAG)
		loudness_bar = self.computeAmplitude(bar, sr, plotflag=self.PLOT_FLAG)
		discreteloudness_bar = self.computeDiscreteFeature(loudness_bar.reshape(1,-1), onsets, sr, plotflag=self.PLOT_FLAG)
		discreteloudness_bar = discreteloudness_bar.reshape(-1)
		centroid_bar = self.computeCentroid(bar, sr, plotflag=self.PLOT_FLAG)
		discretecentroid_bar = self.computeDiscreteFeature(centroid_bar.reshape(1,-1), onsets,sr, plotflag=self.PLOT_FLAG)
		discretecentroid_bar = discretecentroid_bar.reshape(-1)
		flatness_bar = self.computeFlatness(bar, sr, plotflag=self.PLOT_FLAG)
		discreteflatness_bar = self.computeDiscreteFeature(flatness_bar.reshape(1,-1), onsets, sr, plotflag=self.PLOT_FLAG)
		discreteflatness_bar = discreteflatness_bar.reshape(-1)

		sequence_descriptors = [binary_rhythm_bar, onsets, 
								CQT_bar, CQT_center_of_mass_bar, CQT_var_bar,
								chroma_bar, discretechroma_bar,
								loudness_bar, discreteloudness_bar,
								centroid_bar, discretecentroid_bar,
								flatness_bar, discreteflatness_bar,
								tonnetz_bar, discretetonnetz_bar]

		return sequence_descriptors


	# FUNCTIONS TO COMPUTE FEATURES
	def computeBinaryRhythm(self, bar, sr, rhythm_subdivisions=16, plotflag=False):
		# compute STFT
		S_bar = np.abs(librosa.stft(bar))
		# extract onsets
		bar_o_env = librosa.onset.onset_strength(y=bar, sr=sr)
		bar_onset_frames = librosa.onset.onset_detect(onset_envelope=bar_o_env, sr=sr, backtrack=False)
		bar_peaks = librosa.util.peak_pick(bar_o_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=1., wait=0) # adjust delta
		bars_samples = [int(bar.shape[0]/4*i) for i in range(4)]
		
		# plot bar with onsets
		if plotflag:
			fig, ax = plt.subplots(figsize=(6,1))
			ax.plot(bar, label='signal', alpha=0.4)
			ax.vlines(librosa.frames_to_samples(bar_peaks), -0.5,0.5, color='r', alpha=0.7, linestyle='--', label='Onsets')
			ax.vlines(bars_samples, -0.5,0.5, color='r', alpha=0.9, linestyle='-', label='Beats')
			ax.legend()
			ax.set_ylim([-0.5,0.5])
			fig.suptitle('Signal with bar division and rhythmic onsets')
			ax.set_xlabel('Samples')
			ax.set_ylabel('Signal')
			plt.show()

		# compute RMS loudness
		rms = librosa.feature.rms(y=bar)
		rms_at_peaks = [rms[0][i] for i in bar_peaks] 

		# binary rhythm representation
		interval_size = int(rms.shape[1] / rhythm_subdivisions)
		binary_rhythm = []
		dynamic_binary_rhythm = []
		peak_distances = []
		dynamic_counter = 0
		for i in range(0,rms.shape[1],interval_size):
			# if there is a peak in the bar division 1, otherwise 0
			flag = 0
			flag_dynamic = 0
			for peak in bar_peaks:
				if peak > i and peak <= i+interval_size:
					flag = 1
					flag_dynamic = rms_at_peaks[dynamic_counter]
					dynamic_counter += 1
			binary_rhythm.append(flag)
			dynamic_binary_rhythm.append(flag_dynamic)

		'''
		# plot rhythmic subdivisions
		subdivs = list(range(rhythm_subdivisions))
		if plotflag:
			fig, ax = plt.subplots(figsize=(6,1))
			ax.bar(subdivs, dynamic_binary_rhythm, align='center')
			fig.suptitle('Dynamics quantized rhythm')
			ax.set_xlabel('Rhythmic Subdivision')
			ax.set_ylabel('RMS loudness')
			ax.set_ylim([0,0.1])
			plt.show()
		'''

		bar_peaks_times = librosa.frames_to_time(bar_peaks, sr=sr)
		peak_distances = [ float(bar_peaks_times[i]-bar_peaks_times[i-1]) for i in range(1,bar_peaks_times.shape[0]) ]
		bar_peaks = np.insert(bar_peaks, 0, 0, axis=0)

		if plotflag:
			print('Binary rhythm: ', np.array(binary_rhythm))
			#print('Peak distances: ', peak_distances)
			#print('Peak Loudnesses: ', np.array(dynamic_binary_rhythm))
		return np.array(binary_rhythm), bar_peaks, peak_distances, np.array(dynamic_binary_rhythm)

	def computeChroma(self, bar, sr, plotflag=False):
	    S = np.abs(librosa.stft(bar)) # compute STFT
	    chroma = librosa.feature.chroma_stft(S=S, sr=sr) # compute chroma
	    # plot chroma
	    if plotflag:
	        fig, ax = plt.subplots(1, figsize=(5, 1))
	        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
	        fig.colorbar(img)
	        fig.suptitle('Chroma')
	        ax.set_xlabel('Frames')
	        plt.show()
	    return chroma

	def computeTonnetz(self, bar, sr, plotflag=False):
		tonnetz = librosa.feature.tonnetz(y=bar, sr=sr)
		# plot tonnetz
		if plotflag:
			fig, ax = plt.subplots(1, figsize=(5, 1))
			img1 = librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
			ax.set(title='Tonal Centroids (Tonnetz)')
			ax.label_outer()
			plt.show()
		return tonnetz

	def computeDiscreteChroma(self, chroma, onsets, sr, plotflag=False):
		chroma_at_peaks = [chroma[:,i] for i in onsets]
		#bar_peaks = np.append(onsets, [chroma.shape[1]-1], axis=0)	    
		# plot chroma
		if plotflag:
			fig, ax = plt.subplots(1, figsize=(5, 1))
			img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='frames', ax=ax)
			ax.vlines(onsets, 0, 11, color='w', alpha=1, linestyle='-', label='Onsets')
			fig.colorbar(img)
			plt.show()
		return np.array(chroma_at_peaks)

	def computeCQT(self, bar, sr, plotflag=False):
	    CQT = np.abs(librosa.cqt(y=bar, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))
	    CQT_mean = CQT.mean(axis=1)
	    CQT_center_of_mass = sum([CQT_mean[i]*i for i in range(CQT_mean.shape[0])]) / sum(CQT_mean)
	    CQT_var = np.std([CQT_mean[i]*i for i in range(CQT_mean.shape[0])])
	    # plot CQT
	    if plotflag:
	        fig, ax = plt.subplots(1, figsize=(5, 1))
	        img = librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time', ax=ax)
	        ax.set_title('Current bar')
	        fig.colorbar(img)
	        plt.show()
	        # plot avg over y
	        fig, ax = plt.subplots(figsize=(5, 1))
	        ax.vlines(CQT_center_of_mass, CQT_mean.min(), 
	            CQT_mean.max(), color='r', alpha=0.8, linestyle='-', lw=1)
	        ax.vlines([CQT_center_of_mass-CQT_var, CQT_center_of_mass+CQT_var], 
	            CQT_mean.min(), CQT_mean.max(), 
	            color='r', alpha=0.8, linestyle='--', lw=1)
	        ax.axvspan(CQT_center_of_mass-CQT_var, CQT_center_of_mass+CQT_var, ymin=0.1, ymax=0.9, alpha=0.2, color='r')
	        ax.plot(CQT.mean(axis=1), color='r', label='Current bar')
	        ax.set_xlabel("CQT frequency")
	        #ax.set_ylim(-0.1,1.1)
	        ax.set_title('Compare CQT transform averages')
	        ax.legend()
	        plt.show()
	    return CQT, CQT_center_of_mass, CQT_var

	def computeFlatness(self, bar, sr, plotflag=False):
	    flatness = librosa.feature.spectral_flatness(y=bar)
	    # plot flatness
	    if plotflag:
	        fig, ax = plt.subplots(figsize=(5, 1))
	        ax.plot(flatness[0][:], color='b', label='Current bar')
	        fig.suptitle('Continuous spectral flatness')
	        ax.set_xlabel('Frames')
	        ax.set_ylabel('Flatness')
	        plt.show()
	    return flatness

	def computeCentroid(self, bar, sr, plotflag=False):
		#f0, _, _ = librosa.pyin(bar, sr=sr, 
		#						fmin=librosa.note_to_hz('C2'), 
		#						fmax=librosa.note_to_hz('C7'))

		# forward fill nan
		#mask = np.isnan(f0)
		#idx = np.where(~mask,np.arange(mask.shape[0]),0)
		#np.maximum.accumulate(idx, axis=0, out=idx)
		#f0[mask] = f0[idx[mask]]
		#f0 = np.nan_to_num(f0)
		#f0 = f0.reshape(-1,1)

		# computing f0 is too expensive, it is substituted with centroid here
		# as an approxiamtion, however pitch is computed in the live version of the system

		centroid = librosa.feature.spectral_centroid(y=bar, sr=sr)
		# plot centroid
		if plotflag:
			fig, ax = plt.subplots(figsize=(5, 1))
			ax.plot(centroid[0], color='b')
			fig.suptitle('Continuous spectral centroid')
			ax.set_xlabel('Frames')
			ax.set_ylabel('Centroid')
			plt.show()
		return centroid

	def computeAmplitude(self, bar, sr, plotflag=False):
	    amplitude = librosa.feature.rms(y=bar)
	    # plot centroid
	    if plotflag:
	        fig, ax = plt.subplots(figsize=(5, 1))
	        ax.plot(amplitude[0], color='b')
	        fig.suptitle('Continuous RMS loudness')
	        ax.set_xlabel('Frames')
	        ax.set_ylabel('RMS loudness')
	        plt.show()
	    return amplitude

	def computeDiscreteFeature(self, feature, onsets, sr, plotflag=False):
		feature_at_peaks = [feature[:,i] for i in onsets]
		bar_peaks = np.append(onsets, [feature.shape[1]-1], axis=0)	    
		# plot feature
		if plotflag:
			fig, ax = plt.subplots(figsize=(5, 1))
			ax.plot(feature[0], color='b')
			ax.vlines(bar_peaks, feature[0].min(), feature[0].max(), color='r', alpha=0.7, linestyle='--', label='Onsets')
			for pitch in range(len(feature_at_peaks)):
				ax.hlines(feature_at_peaks[pitch], xmin=bar_peaks[pitch] , xmax=bar_peaks[pitch+1], color='r', alpha=0.8, linestyle='-')
			fig.suptitle('Discrete feature')
			ax.set_xlabel('Frames')
			ax.set_ylabel('Feature')
			plt.show()
		return np.array(feature_at_peaks)


	# FUNCTIONS TO COMPARE FEATURES
	def compareBinaryRhythms(self, binary_rhythm1, binary_rhythm2, rhythm_subdivisions=16):
		if self.verbose >= 1:
			print(np.array(binary_rhythm1))
			print(np.array(binary_rhythm2))
		binary_comparison = [1 if binary_rhythm1[i] == binary_rhythm2[i] else 0 for i in range(len(binary_rhythm1))]
		binary_comparison_coefficient = sum(binary_comparison) / rhythm_subdivisions 
		if self.verbose >= 1:
			print(f'Binary Comparison coefficient: {binary_comparison_coefficient:.3f}')

		rhythm_density_coefficient = abs(np.array(binary_rhythm1).sum() - np.array(binary_rhythm2).sum()) / rhythm_subdivisions
		if self.verbose >= 1:
			print(f"Rhythm Density Comparison coefficient: {rhythm_density_coefficient:.3f}")
		return binary_comparison_coefficient, rhythm_density_coefficient

	def compareSpectralBandwidth(self, CQT1, CQT1_center_of_mass, CQT1_var, CQT2, CQT2_center_of_mass, CQT2_var, plotflag=False):

		CQT1_mean = CQT1.mean(axis=1)
		CQT2_mean = CQT2.mean(axis=1)

		if plotflag:
			# plot comparision of CQT-spectral bandwidth
			fig, ax = plt.subplots(figsize=(5, 1))
			ax.vlines(CQT1_center_of_mass, CQT1_mean.min(), 
			    CQT1_mean.max(), color='r', alpha=0.8, linestyle='-', lw=1)
			ax.vlines([CQT1_center_of_mass-CQT1_var, CQT1_center_of_mass+CQT1_var], 
			    CQT1_mean.min(), CQT1_mean.max(), 
			    color='r', alpha=0.8, linestyle='--', lw=1)
			ax.axvspan(CQT1_center_of_mass-CQT1_var, CQT1_center_of_mass+CQT1_var, ymin=0.1, ymax=0.9, alpha=0.2, color='r')
			ax.plot(CQT1_mean, color='r', label='Bar 1')

			ax.vlines(CQT2_center_of_mass, CQT2_mean.min(), CQT2_mean.max(), color='b', alpha=0.8, linestyle='-', lw=1)
			ax.vlines([CQT2_center_of_mass-CQT2_var, CQT2_center_of_mass+CQT2_var], CQT2_mean.min(), CQT2_mean.max(), 
			    color='b', alpha=0.8, linestyle='--', lw=1)
			ax.axvspan(CQT2_center_of_mass-CQT2_var, CQT2_center_of_mass+CQT2_var, ymin=0.1, ymax=0.9, alpha=0.2, color='r')
			ax.plot(CQT2_mean, color='b', label='Bar 2')

			ax.plot(np.abs(CQT1_mean - CQT2_mean), color='g', label='Difference')

			ax.set_xlabel("CQT frequency")
			#ax.set_ylim(-0.1,1.1)
			ax.set_title('Compare CQT transform averages')
			ax.legend()
			plt.show()

		spectral_energy_overlap_index = max(0, 
		    min(CQT1_center_of_mass+CQT1_var, CQT2_center_of_mass+CQT2_var) - max(CQT1_center_of_mass-CQT1_var, CQT2_center_of_mass-CQT2_var))
		spectral_energy_overlap_coefficient = min(spectral_energy_overlap_index, min(CQT1_var*2, CQT2_var*2)) / min(CQT1_var*2, CQT2_var*2)
		if self.verbose >= 1:
			print(f"Spectral energy overlap coefficient: {spectral_energy_overlap_coefficient:.3f}")

		spectral_energy_difference_coefficient = np.abs(CQT1_mean - CQT2_mean).mean()
		if self.verbose >= 1:
			print(f"Spectral energy difference coefficient: {spectral_energy_difference_coefficient:.3f}")
		return spectral_energy_overlap_coefficient, spectral_energy_difference_coefficient

	def computeSignalsMSE(self, signal1, signal2):
		minsignal = min(signal1.min(), signal2.min())
		maxsignal = max(signal1.max(), signal2.max())
		normalized_signal1 = (signal1 - minsignal) / (maxsignal - minsignal)
		normalized_signal2 = (signal2 - minsignal) / (maxsignal - minsignal)
		MSE = ((normalized_signal1 - normalized_signal2)**2).mean()
		if self.verbose >= 1:
			print(f'MSE between the two signal is: {MSE:.3f}')
		return MSE

	def computeContinuousCorrelation(self, signal1, signal2, plotflag=False):

		if (signal1.max() - signal1.min()) != 0 and (signal2.max() - signal2.min()) != 0:

			normalized_signal1 = (signal1 - signal1.min()) / (signal1.max() - signal1.min())
			normalized_signal2 = (signal2 - signal2.min()) / (signal2.max() - signal2.min())
			signal_time_correlation = normalized_signal1 * normalized_signal2

			if plotflag:
				fig, ax = plt.subplots(figsize=(5, 1))
				ax.plot(normalized_signal1[0], color='b', label='signal 1')
				ax.plot(normalized_signal2[0], color='r', label='signal 2')
				ax.plot(signal_time_correlation[0], color='g', label='correlation')
				fig.suptitle('Continuous signals correlation')
				ax.set_xlabel('frames')
				ax.set_ylabel('signal')
				ax.legend()
				plt.show()

			time_correlation_coefficient = np.mean(signal_time_correlation.sum(axis=1) / signal_time_correlation.shape[1])
			#print(f"Time correlation coefficient: {time_correlation_coefficient:.3f}")
			pearson_correlation = ((signal1 - signal1.mean()) * (signal2 - signal2.mean()) / (signal1.std() * signal2.std())).mean()

			if np.isnan(time_correlation_coefficient):
				time_correlation_coefficient = 0
			if np.isnan(pearson_correlation):
				pearson_correlation = 0

			if self.verbose >= 1:
				print(f"Continuous pearson correlation coefficient: {pearson_correlation:.3f}")
		else:
			time_correlation_coefficient = 0
			pearson_correlation = 0
			if self.verbose >= 1:
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

			if plotflag:
				fig, ax = plt.subplots(figsize=(5, 1))
				ax.step(both_onsets, normalized_values1, where='post', color='r', label='signal 1')
				ax.step(both_onsets, normalized_values2, where='post', color='b', label='signal 1')
				ax.step(both_onsets, normalized_values_time_correlation, where='post', color='g', label='correlation')
				fig.suptitle('Discrete signals correlation')
				ax.set_xlabel('frames')
				ax.set_ylabel('values')
				ax.legend()
				plt.show()

			discrete_time_correlation_coefficient = np.mean(normalized_values_time_correlation.sum() / normalized_values_time_correlation.shape[0])
			#print(f"Discrete time correlation coefficient: {discrete_time_correlation_coefficient:.3f}")
			discrete_pearson_correlation = ((new_values1 - new_values1.mean()) * (new_values2 - new_values2.mean()) / (new_values1.std() * new_values2.std())).mean()

			if np.isnan(discrete_time_correlation_coefficient):
				discrete_time_correlation_coefficient = 0
			if np.isnan(discrete_pearson_correlation):
				discrete_pearson_correlation = 0

			if self.verbose >= 1:
				print(f"Discrete pearson correlation coefficient: {discrete_pearson_correlation:.3f}")
		else: 
			discrete_time_correlation_coefficient = 0
			discrete_pearson_correlation = 0

		return discrete_time_correlation_coefficient, discrete_pearson_correlation

	def computeTwodimensionalMSE(self, signal1, signal2):
		minsignal = np.concatenate((signal1.min(axis=1).reshape(-1,1), signal2.min(axis=1).reshape(-1,1)), axis=1).min(axis=1).reshape(-1,1)
		maxsignal = np.concatenate((signal1.max(axis=1).reshape(-1,1), signal2.max(axis=1).reshape(-1,1)), axis=1).max(axis=1).reshape(-1,1)
		normalized_signal1 = (signal1 - minsignal) / (maxsignal - minsignal)
		normalized_signal2 = (signal2 - minsignal) / (maxsignal - minsignal)
		MSE = ((normalized_signal1 - normalized_signal2)**2).mean(axis=1).mean()
		if self.verbose >= 1:
			print(f'MSE between the two signal is: {MSE:.3f}')
		return MSE

	def computeTwodimensionalAE(self, signal1, signal2, plotflag=False):
		if plotflag:
			fig, ax = plt.subplots(1, figsize=(5, 1))
			ax.plot(signal1.mean(axis=1), label="signal 1")
			ax.plot(signal2.mean(axis=1), label="signal 2")
			plt.legend()
			plt.show()

		similarity_coefficient = np.abs(np.mean(signal1.mean(axis=1) - signal2.mean(axis=1)))
		if self.verbose >= 1:
			print(f"Absolute Error difference coefficient: {similarity_coefficient:.3f}")
		return similarity_coefficient

	def computeTwodimensionalContinuousCorrelation(self, signal1, signal2, plotflag=False):
		time_correlation = signal1 * signal2
		time_correlation_coefficient = np.mean(time_correlation.sum(axis=1) / time_correlation.shape[1])
		
		if plotflag:
			INDEX = 0
			fig, ax = plt.subplots(1, figsize=(5, 1))
			ax.plot(signal1[INDEX,:], label="chroma bar 1")
			ax.plot(signal2[INDEX,:], label="chroma bar 2")
			ax.plot(time_correlation[INDEX,:], label="correlation")
			plt.legend()
			fig.suptitle('Correlation of chroma value for note A')
			plt.show()
		#print(f"Continuous time correlation coefficient: {time_correlation_coefficient:.3f}")

		continuous_pearson_correlation = ((signal1 - signal1.mean(axis=1).reshape(-1,1)) * (signal2 - signal2.mean(axis=1).reshape(-1,1))).mean(axis=1) / (signal1.std(axis=1) * signal2.std(axis=1))

		if np.isnan(np.sum(time_correlation_coefficient)):
			time_correlation_coefficient = np.array([0])
		if np.isnan(np.sum(continuous_pearson_correlation)):
			continuous_pearson_correlation = np.array([0])

		if self.verbose >= 1:
			print(f"Continuous pearson correlation coefficient: {continuous_pearson_correlation.mean():.3f}")
		return time_correlation_coefficient, continuous_pearson_correlation.mean()

	def computeTwodimensionalDiscreteCorrelation(self, onsets1, values1, onsets2, values2):

		if onsets1.shape[0] == 0 or onsets2.shape[0] == 0:
			
			discrete_time_correlation_coefficient = 0
			discrete_pearson_correlation = 0

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
				#print(f"Discrete time correlation coefficient: {discrete_time_correlation_coefficient:.3f}")

				discrete_pearson_correlation = ((new_values1 - new_values1.mean(axis=1).reshape(-1,1)) * (new_values2 - new_values2.mean(axis=1).reshape(-1,1))).mean(axis=1) / (new_values1.std(axis=1) * new_values2.std(axis=1))

				if np.isnan(np.sum(discrete_time_correlation_coefficient)):
					discrete_time_correlation_coefficient = np.array([0])
				if np.isnan(np.sum(discrete_pearson_correlation)):
					discrete_pearson_correlation = np.array([0])

			except:
				if self.verbose >= 1:
					print('COULD NOT COMPUTE CORRELATION')
				discrete_time_correlation_coefficient = np.array([0])
				discrete_pearson_correlation = np.array([0])

			if self.verbose >= 1:
				print(f"Discrete pearson correlation coefficient: {discrete_pearson_correlation.mean():.3f}")

		return discrete_time_correlation_coefficient, discrete_pearson_correlation.mean()


if __name__ == '__main__': 

	## DEFINE SCRIPT PARAMETERS
	soundfile_filepath = './00_corpus/USE CASE 1.wav'
	config_filepath = './config.json'
	output_dir_path = "./01_output_offline"
	looper = AutonomousLooperOffline(soundfile_filepath, config_filepath=config_filepath, plotFlag=False)
	looper.computeLooperTrack(output_dir_path)

