import os
from rich import print
from rich.console import Console
console = Console()

data_dirs = ["cts_rabi_amp_0",
             "cts_rabi_amp_1",
             "cts_rabi_amp_2",
             "cts_rabi_amp_3",
             "cts_rabi_amp_4",
             "cts_rabi_amp_5"]

analyze_dirs = ["200517_160211_cts_rabi_amp_0_prep_Y_all_times",
                "200518_151250_cts_rabi_amp_1_prep_Y_all_times",
                "200518_152413_cts_rabi_amp_2_prep_Y_all_times",
                "200518_150037_cts_rabi_amp_3_prep_Y_all_times",
                "200517_171648_cts_rabi_amp_4_prep_Y_all_times",
                "200518_112931_cts_rabi_amp_5_prep_Y_all_times"]

analysis_script_path = r"/home/qnl/Git-repositories/machine_learning_test/scripts/single_prep_multi_timestep_analyze.py"

for data_dir, analyze_dir in zip(data_dirs, analyze_dirs):
    console.print(f"Processing data from {data_dir}", style="bold red")
    datapath = os.path.join(r"/home/qnl/Git-repositories/machine_learning_test/data", data_dir, "prep_Y")
    filepath = os.path.join(r"/home/qnl/Git-repositories/machine_learning_test/analysis/rabi_amp_sweep", analyze_dir)
    exec(open(analysis_script_path).read())