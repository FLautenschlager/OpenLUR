from subprocess import call
import sys

models = [
    "AutoML.py",
    "RF_autosearch.py",
    "RF_randomsearch.py"
]
seasons = [1,2,3,4]
refits = [True, False]
features = [1,3]

for season in [1,2,3,4]:
    for model in models:
        for refit in refits:
            for feature in features:
                if model=="RF_autosearch.py":
                    p=10
                else:
                    p=40
                command = ["sbatch -p ocropus -c 40 run_slurm.sh", model, "-f" ,feature, "-n", season, "-i", 40, "-p", p, "-t", sys.argv[1]]
                if refit:
                    command.append("-r")
                print(command)
                call(command)

