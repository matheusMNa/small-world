#!/bin/bash
# Job array do ensemble de vacinação: uma tarefa por (configuração, pedaço de seeds).
#
# Uso, a partir da raiz do repositório:
#   mkdir -p logs results/pedacos
#   sbatch cluster/ensemble_array.slurm
#   # quando o array terminar:
#   python analyses/juntar_pedacos.py
#   python analyses/figuras_vacinacao.py
#
# O intervalo de --array precisa bater com o tamanho do pedaço. Com
# TAMANHO_PEDACO=500 cada configuração é uma tarefa só: 52 tarefas, índices 0 a 51
# (0-14: cenários com bloco central; 15-29: espalhados; 30-39: semente inicial;
# 40-51: conexões globais). Para outro tamanho, ou para ver o índice de cada
# configuração, use
#   python analyses/ensemble.py --tamanho-pedaco <P> --listar
# O "%16" limita a 16 tarefas simultâneas; ajuste ao número de CPUs disponíveis.
# Relançar o mesmo sbatch é seguro: pedaços já gravados são pulados.
# Para rodar só algumas tarefas: sbatch --array=30-51 cluster/ensemble_array.slurm

#SBATCH --job-name=small-world-vac
#SBATCH --array=0-51%16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --mem=4G
#SBATCH --time=10:00:00
# Partição, memória e tempo são os usados no cluster do grupo; em outro cluster,
# confira com sinfo. Referência da primeira rodada (15 cenários, 500 seeds, 1500
# passos, uma CPU por tarefa): de 5 a 68 minutos por tarefa e cerca de 240 MB de
# memória. O tempo variou até 4 vezes entre tarefas parecidas, conforme a
# ocupação do nó, então o limite de tempo é folgado de propósito: um pedaço só é
# gravado quando a tarefa termina.
#SBATCH --output=logs/ensemble_%A_%a.out
#SBATCH --error=logs/ensemble_%A_%a.err

set -euo pipefail

TAMANHO_PEDACO=500

# Uma thread por processo: o paralelismo vem do array, não de BLAS/OpenMP.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ambiente Python. Em outro cluster, troque estas linhas pelo que ele usa
# (por exemplo, "module load <módulo do conda>" antes de ativar).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate small-world
cd "${SLURM_SUBMIT_DIR:-.}"

# Fica registrado no log com que versões os resultados foram gerados.
python -c "import sys, numpy, scipy; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__)"

python analyses/ensemble.py \
    --tarefa "${SLURM_ARRAY_TASK_ID}" \
    --tamanho-pedaco "${TAMANHO_PEDACO}" \
    --processos "${SLURM_CPUS_PER_TASK:-1}" \
    --saida results/pedacos
