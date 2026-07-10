

config=./data_new/inclv10_aux/ak15_MD_inclv10_xggg_finetune_hybridv2_tau.yaml
PREFIX=GloParT_v2_finetune_xggg_hybridv2_tau_fc_512_0.1_512_0.1_suff_fc0_512_0.1_512_0.1_lr_1e-6_mult100_20_s7
# 预测 ROOT 写到数据盘（覆盖 aux 脚本里的 predict/$PREFIX/pred.root）
PREDICT_ROOT_BASE=/aifs/user/data/kouzx/datasets/predict

#test --predict --model-prefix $test_epoch
freeze='^((main|module\.main)\.).*'
regexmatch='^(?!(main|module\.main)\.).*' # match the parameters that do not start with "main."

#--freeze-model-weights $freeze --start-lr 1e-3 \
#--optimizer-option lr_mult (\"$regexmatch\",1000) --start-lr 1e-6 \
modelftopts="--network-config networks/example_ParticleTransformer2024PlusTagger_unified2_hybrid.py \
-o finetune_kw {'freeze_main_params':False,'mode':'hybrid','target_inds':None,'target_reg_inds':[314,315],'num_ft_nodes':5,'num_ft_cls_nodes':3,'num_ft_reg_nodes':2,'fc_params':[(512,0.1),(512,0.1)],'fc_suff_kw':{'append_after':'fc.0','params':[(512,0.1),(512,0.1)]}} \
-o label_cls_nodes ['label_xggg','label_top','label_qcd'] -o label_stored ['label_xggg','label_top','label_qcd'] \
--load-model-weights finetune_stage3_adaptstage2 \
--optimizer-option lr_mult (\"$regexmatch\",100) --start-lr 1e-6 \
--optimizer-option weight_decay 0.01
--train-mode hybrid --num-epochs 20 "

modelopts="-o num_nodes 316 -o num_cls_nodes 314 -o use_swiglu_config False -o use_pair_norm_config False \
-o fc_params [(1024,0.1)] -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 -o num_layers 8 \
-o reg_kw {'gamma':5.} -o adapt_stage2_model True " # stage-2 config compatible to stage-3 model file

data_dir=/afs/ihep.ac.cn/users/k/kouzx/publicfs/dnntuples/v7plus_ak15

#xggg:${data_dir}/BulkGravitonToHHTo6Glu_MX-600to6000_MH-15to250/*.root \
#xggghm:${data_dir}/BulkGravitonToHHTo6Glu_MX-Var_MH-260to650/*.root \
#xggg_lowpt:${data_dir}/BulkGravitonToHHTo6Glu_MX-Var_MH-15to650_LowPt/*.root \
#top:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root \
#tophm:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-Var_MH-260to650/*.root \
#top_lowpt:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-Var_MH-15to6250_LowPt/*.root \
#qcd:${data_dir}/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root \
trainvalopts="--run-mode train --load-epoch 1 \
--data-train \
xggg:${data_dir}/BulkGravitonToHHTo6Glu_MX-600to6000_MH-15to250/*.root \
top:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root \
qcd:${data_dir}/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root \
--num-workers 8 --fetch-step 1. --data-split-num 100 --data-split-num-val 300 \
--samples-per-epoch $((5000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1250 * 1024 / $NGPUS)) "

# xgg:${data_dir}/BulkGravitonToHHTo6Glu_MX-600to6000_MH-15to250_infer/*.root \
# xggg:${data_dir}/BulkGravitonToHHTo6Glu_MX-600to6000_MH-15to250_infer/*.root \
# top:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250_infer/*.root \
# qcd:${data_dir}/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8_infer/*.root \
# ZRggg_M100_AN:${data_dir}/WZR_012j_NLO_WToLNu_ZRTo3Glu_M-100_13TeV-pythia8_infer/*.root \
# ZRggg_M250_AN:${data_dir}/WZR_012j_NLO_WToLNu_ZRTo3Glu_M-250_13TeV-pythia8_infer/*.root \
# ttbar_AN:${data_dir}/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_infer/*.root \
# wjets600_AN:${data_dir}/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
# wjets400_AN:${data_dir}/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
# wjets250_AN:${data_dir}/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
# wjets100_AN:${data_dir}/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
testopts="--run-mode test --num-workers 3 --data-split-num 1 \
--model-prefix model/${PREFIX}/net_best_epoch_state.pt \
--data-test \
xggg:${data_dir}/BulkGravitonToHHTo6Glu_MX-600to6000_MH-15to250_infer/*.root \
top:${data_dir}/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250_infer/*.root \
qcd:${data_dir}/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8_infer/*.root \
ZRggg_M100_AN:${data_dir}/WZR_012j_NLO_WToLNu_ZRTo3Glu_M-100_13TeV-pythia8_infer/*.root \
ZRggg_M250_AN:${data_dir}/WZR_012j_NLO_WToLNu_ZRTo3Glu_M-250_13TeV-pythia8_infer/*.root \
ttbar_AN:${data_dir}/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_infer/*.root \
wjets600_AN:${data_dir}/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
wjets400_AN:${data_dir}/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
wjets250_AN:${data_dir}/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
wjets100_AN:${data_dir}/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_infer/*.root \
--predict-output ${PREDICT_ROOT_BASE}/${PREFIX}/pred.root \
 "

source scripts/aux/train_GloParT_v3beta4_hgamgam_finetune.sh run 0,1 --batch-size 512 $modelopts $modelftopts $trainvalopts
<<1
python train.py --gpus 0 --export-onnx ./model/${PREFIX}/model_opset14.onnx \
--data-config ${config} --network-config networks/example_ParticleTransformer2024PlusTagger_unified2_hybrid.py \
--model-prefix model/${PREFIX}/net_best_epoch_state.pt $modelopts
<<1
source scripts/aux/train_GloParT_v3beta4_hgamgam_finetune.sh dryrun 0 --batch-size 2048 --start-lr 4e-3 $modelopts $modelftopts $testopts
