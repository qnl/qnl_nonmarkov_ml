import os
data_dirs = ["cts_rabi_amp_2"]#,
             #"cts_rabi_amp_1",
             #"cts_rabi_amp_2",
             #"cts_rabi_amp_3",
             #"cts_rabi_amp_4",
             #"cts_rabi_amp_5"]

script_path = r"/home/qnl/Git-repositories/machine_learning_test/scripts/single_prep_multi_timestep.py"
analysis_script_path = r"/home/qnl/Git-repositories/machine_learning_test/scripts/single_prep_multi_timestep_analyze.py"

for data_dir in data_dirs:
    filepath = os.path.join(r"/home/qnl/Git-repositories/machine_learning_test/data", 
			      data_dir, "prep_Y")
    exec(open(script_path).read())
    exec(open(analysis_script_path).read())
