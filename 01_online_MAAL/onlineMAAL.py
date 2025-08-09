import sys
import argparse
import os

# import main script from parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onlineMAALclass import AutonomousLooperOnline

if __name__ == '__main__': 

	parser = argparse.ArgumentParser()
	parser.add_argument('--CONFIG_FILEPAHT', type=str, default='./01_online_MAAL/config.json',
						help='path to the configuration file')
	parser.add_argument('--MAAL_PD_FILEPAHT', type=str, default='./01_online_MAAL/00_MAAL_PD/_main.pd',
						help='path to the PD MAAL patch')
	args = parser.parse_args(sys.argv[1:])

	config_filepath = args.CONFIG_FILEPAHT
	pd_looper_path = args.MAAL_PD_FILEPAHT
	looper = AutonomousLooperOnline(config_filepath=config_filepath, pd_looper_path=pd_looper_path)
