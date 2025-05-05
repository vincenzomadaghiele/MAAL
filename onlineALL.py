import sys
import argparse
from onlineALLclass_v2 import AutonomousLooperOnline

if __name__ == '__main__': 

	parser = argparse.ArgumentParser()
	parser.add_argument('--CONFIG_FILEPAHT', type=str, default='./config.json',
						help='path to the configuration file')
	args = parser.parse_args(sys.argv[1:])

	config_filepath = args.CONFIG_FILEPAHT
	looper = AutonomousLooperOnline(config_filepath=config_filepath)
