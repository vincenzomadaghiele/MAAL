import sys
import argparse
from offlineALLclass import AutonomousLooperOffline

if __name__ == '__main__': 

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--SOUNDFILE_FILEPATH', type=str, default='./00_corpus/USE CASE 2.wav',
						help='name of the folder containing the soundfile')
	parser.add_argument('--CONFIG_FILEPAHT', type=str, default='./config.json',
						help='path to the configuration file')
	parser.add_argument('--OUTPUT_DIR_PATH', type=str, default="./01_output_offline",
						help='path to the output file')
	args = parser.parse_args(sys.argv[1:])

	## DEFINE SCRIPT PARAMETERS
	soundfile_filepath = args.SOUNDFILE_FILEPATH
	config_filepath = args.CONFIG_FILEPAHT
	output_dir_path = args.OUTPUT_DIR_PATH

	looper = AutonomousLooperOffline(soundfile_filepath, config_filepath=config_filepath, plotFlag=False)
	looper.computeLooperTrack(output_dir_path)
