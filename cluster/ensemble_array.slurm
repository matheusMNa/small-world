#!/bin/bash
# Job array do ensemble de vacinação: uma tarefa por (cenário, pedaço de seeds).
#
# ATENÇÃO: este script não foi testado em nenhum cluster; só a sintaxe foi
# conferida (bash -n). Preencha os campos marcados com PREENCHER antes de usar.
#
# Uso, a partir da raiz do repositório:
#   mkdir -p logs results/pedacos
#   sbatch cluster/ensemble_array.slurm
#   # quando o array terminar:
#   python analyses/juntar_pedacos.py
#   python analyses/figuras_vacinacao.py
#
# O intervalo de --array precisa bater com o tamanho do pedaço. Com
# TAMANHO_PEDACO=500 cada cenário é uma tarefa só: 15 tarefas, índices 0 a 14.
# Para outro tamanho, confira o número de tarefas com
#   python analyses/ensemble.py --tamanho-pedaco <P> --listar
# Relançar o mesmo sbatch é seguro: pedaços já gravados são pulados.

#SBATCH --job-name=small-world-vac
#SBATCH --array=0-14
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=PREENCHER_PARTICAO
#SBATCH --mem=PREENCHER_MEMORIA
#SBATCH --time=PREENCHER_TEMPO
# Referência para preencher. É uma estimativa, não uma medida no cluster: uma
# execução Flu-like de 1500 passos levou de 1,8 a 2,3 s em outra CPU, o que dá
# de 15 a 20 minutos por tarefa com TAMANHO_PEDACO=500 (500 execuções em um núcleo).
# Os arrays de um pedaço de 500 seeds somam algumas centenas de MB. Algo como
# --mem=2G e --time=01:00:00 deve deixar folga; ajuste à CPU do cluster.
#SBATCH --output=logs/ensemble_%A_%a.out
#SBATCH --error=logs/ensemble_%A_%a.err

set -euo pipefail

TAMANHO_PEDACO=500

# Uma thread por processo: o paralelismo vem do array, não de BLAS/OpenMP.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# PREENCHER: carregue aqui o ambiente Python do cluster, por exemplo
#   module load <módulo do conda>
#   conda activate small-world

cd "${SLURM_SUBMIT_DIR:-.}"

python analyses/ensemble.py \
    --tarefa "${SLURM_ARRAY_TASK_ID}" \
    --tamanho-pedaco "${TAMANHO_PEDACO}" \
    --processos "${SLURM_CPUS_PER_TASK:-1}" \
    --saida results/pedacos
