def generate_slurm_script(N, num_iters, train_steps, benchmark_size):
    filename = f"eval_N={N}_iters={num_iters}.slurm"
    content = f"""#!/bin/bash
#SBATCH --job-name=eval_N={N}_iters={num_iters}   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=rh8490@princeton.edu

module purge
module load anaconda3/2024.10
conda activate myenv

python TDTSP_trainer.py {N} {num_iters} {train_steps} {benchmark_size}
"""
    with open(filename, "w") as f:
        f.write(content)

if __name__ == "__main__":
    num_iters = 50
    train_steps = 500_000
    benchmark_size = 100

    for N in [10, 20, 30, 40, 100]:
        generate_slurm_script(N, num_iters, train_steps, benchmark_size)