from subprocess import call
import sys

models = [
    "AutoML.py",
    "RF_autosearch.py",
    "RF_randomsearch.py"
]
seasons = [1,2,3,4]
refits = [
        #False, 
        True
        ]
features = [
        #1,
        3
        ]

for season in [1,2,3,4]:
    for model in models:
        for refit in refits:
            for feature in features:
                if "auto" in model.lower():
                    p=5
                else:
                    p=40
                command = ["/usr/bin/sbatch", "-p", "dmir", "-c", "40", "run_slurm.sh", str(model), "-f" ,str(feature), "-n", str(season), "-i", "40", "-p", str(p), "-t", str(sys.argv[1])]
                if refit:
                    command.append("-r")
                print(command)
                call(command)

