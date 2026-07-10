#!/bin/bash
# 纯 CPU：用全量训练文件生成 .auto.yaml（standardization + reweight hists）
# 不需要 GPU，可在登录节点 / CPU 队列单独跑，再开双卡训练。
#
# 用法（在 weaver-core/weaver 目录下）:
#   bash gen_xggg_auto.sh              # 已有 auto 则跳过
#   REMAKE_AUTO=1 bash gen_xggg_auto.sh  # 强制重算
# 日志写入 logs/yaml_auto/<yaml名>_YYYYMMDD_HHMMSS.log（含 reweight 矩阵打印）

set -euo pipefail

current_dir=$(pwd)
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

config=./data_new/inclv10_aux/ak15_MD_inclv10_xggg_finetune.yaml
data_dir=/afs/ihep.ac.cn/users/k/kouzx/publicfs/dnntuples/v7plus_ak15

DATATRAIN=(
    "xggg:${data_dir}/BulkGravitonToHHTo6Glu_MX-600to6000_MH-15to250/*.root"
    "xggghm:${data_dir}/BulkGravitonToHHTo6Glu_MX-Var_MH-260to650/*.root"
    "xggg_lowpt:${data_dir}/BulkGravitonToHHTo6Glu_MX-Var_MH-15to650_LowPt/*.root"
    "top:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root"
    "tophm:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-Var_MH-260to650/*.root"
    "top_lowpt:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-Var_MH-15to6250_LowPt/*.root"
    "qcd:${data_dir}/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root"
)

AUTO_YAML=$(python -c "
import hashlib
p='${config}'
h=hashlib.md5()
with open(p,'rb') as f:
    for c in iter(lambda:f.read(4096), b''):
        h.update(c)
print(p.replace('.yaml', '.'+h.hexdigest()+'.auto.yaml'))
")

LOG_DIR=logs/yaml_auto
mkdir -p "${LOG_DIR}"
YAML_BASENAME=$(basename "${config}" .yaml)
LOG_FILE="${LOG_DIR}/${YAML_BASENAME}_$(date +%Y%m%d_%H%M%S).log"

extra=()
if [ "${REMAKE_AUTO:-0}" = "1" ]; then
    extra+=(--remake-weights)
fi

{
    echo "data-config: ${config}"
    echo "auto-yaml:   ${AUTO_YAML}"
    echo "log-file:    ${LOG_FILE}"
    if [ "${REMAKE_AUTO:-0}" = "1" ]; then
        echo "REMAKE_AUTO=1: will recompute even if auto exists"
    fi
    echo

    # 显式不用 GPU；_configLogger(stdout) 后 reweight 矩阵会进 stdout，由 tee 落盘
    CUDA_VISIBLE_DEVICES= python scripts/gen_data_auto.py \
        --data-config "${config}" \
        --data-train "${DATATRAIN[@]}" \
        "${extra[@]}"

    echo
    echo "Done. Use this auto file in multi-GPU training (ranks will load it automatically)."
} 2>&1 | tee "${LOG_FILE}"

echo "Log saved to: ${LOG_FILE}"
