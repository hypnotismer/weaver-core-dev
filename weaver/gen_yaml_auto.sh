#!/bin/bash
# 纯 CPU：用全量训练文件生成 .auto.yaml（standardization + reweight hists）
# 不需要 GPU，可在登录节点 / CPU 队列单独跑，再开双卡训练。
#
# 用法（在 weaver-core/weaver 目录下）:
#   bash gen_yaml_auto.sh                # 已有 auto 则跳过
#   REMAKE_AUTO=1 bash gen_yaml_auto.sh  # 强制重算
# 日志写入 logs/yaml_auto/<yaml名>_YYYYMMDD_HHMMSS.log（含 reweight 矩阵打印）
#
# 注意：sculpt yaml 含 reweight_basewgt，WeightMaker 会读全量文件（非 chunked），首次较慢。

set -euo pipefail

current_dir=$(pwd)
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

config=./data_new/inclv10_aux/ak15_MD_inclv10_xggg_finetune_m20-360_hybrid_sculpt.yaml
data_dir=/afs/ihep.ac.cn/users/k/kouzx/publicfs/dnntuples/v7plus_ak15

# 与 train_xggg_sculpt.sh 的 --data-train 保持一致（含 high-mass + AN sculpt 样本）
DATATRAIN=(
    "xggg:${data_dir}/BulkGravitonToHHTo6Glu_MX-600to6000_MH-15to250/*.root"
    "xggghm:${data_dir}/BulkGravitonToHHTo6Glu_MX-Var_MH-260to650/*.root"
    "top:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root"
    "tophm:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-Var_MH-260to650/*.root"
    "qcd:${data_dir}/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root"
    "ttbar:${data_dir}/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*.root"
    "wjets600:${data_dir}/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/*.root"
    "wjets400:${data_dir}/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/*.root"
    "wjets250:${data_dir}/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/*.root"
    "wjets100:${data_dir}/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/*.root"
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
    echo "data-train:"
    for e in "${DATATRAIN[@]}"; do
        echo "  ${e}"
    done
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
