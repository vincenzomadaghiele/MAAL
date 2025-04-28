from offlineALLclass import AutonomousLooperOffline


if __name__ == '__main__': 

	soundfile_filepath = '00_corpus/USE CASE 1.wav'
	config_filepath = './config.json'
	output_dir_path = './01_output_offline'
	looper = AutonomousLooperOffline(soundfile_filepath, config_filepath=config_filepath, plotFlag=False)
	looper.computeLooperTrack(output_dir_path)

