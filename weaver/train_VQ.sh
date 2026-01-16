# model options can override existing options in the run script
DATADIR=/publicfs/cms/user/licq/datasets/JetClassII # ihep
export CUPY_CACHE_DIR="~/mlgpu/.cache/cupy"
mkdir -p "$CUPY_CACHE_DIR"
# Sophon-VQVAE model (vqtorch VectorQuant codebook; optional kmeans_init warmup)
config=./data_pheno/JetClassII_v2/JetClassII_full_VQVAE_nonscale_manual.yaml
PREFIX=Sophon-VQVAE_official_logits1_vq1_pf_1_gen_1_codebook_dim_64_size_64_180
PREDICT=./predict

train_codebook='((mod|module\.mod)\.|mod_fc\.|mod_proj\.).*'
modelopts="--network-config networks/pheno2/example_Sophon_VQVAE.py \
--freeze-model-weights $train_codebook \
-o vq_kw {'logits_weight':0,'vq_weight':1.0,'loss_pf_align_weight':0,'loss_gen_align_weight':1.0,'codebook_dim':16,'codebook_size':1024,'enc_proj':[],'gen_proj':[],'qcd_label_range':(161,188),'entropy_reg':1,'kmeans_init':True,'kmeans_warmup_steps':20,'sync_nu':2.0,'affine_lr':10.0,'affine_groups':1,'beta':0.9,'vq_loss_update_freq':4,'replace_freq':10} \
"
#test --predict --model-prefix $test_epoch

trainvalopts="--run-mode train \
--num-epochs 20 --num-workers 3 --fetch-step 1. --data-split-num 400 \
--samples-per-epoch $((1000 * 8192 )) --samples-per-epoch-val $((2500 * 2048)) "
#40,000 batch
source scripts/train_Sophon_v1_noqcd.sh dryrun 0 --batch-size 8192 --start-lr 5e-4 $modelopts $trainvalopts
1

#bb_deltaR with Sophon-CLIP
<<1
PREFIX=bb_deltaR_Sophon-VQVAE_official_logits1_vq1_ema_codebook_dim64_size32_180_9_512_0.1_512_0.1_1024_1e-3_1
config=./data_pheno/JetClassII_v2/JetClassII_full_nonscale_finetune_bbdeltaR.yaml
test_epoch=./model/${PREFIX}/net_epoch-19_state.pt

modelpath="/afs/ihep.ac.cn/users/k/kouzx/mlgpu/weaver-core/weaver/model/Sophon-VQVAE_official_logits1_vq1_ema_codebook_dim64_size32_180/net_epoch-9_state.pt"

freeze='(mod|module\.mod)\.(?!(cls_token|fc)).*'
exclude='(((gen|module\.gen)|(mod|module\.mod)\.(fc)|(mod_.*|module\.mod_.*)\.(fc)).*)|(mod|module\.mod)\.(cls_token)|(mod|module\.mod)\.(cls_blocks_aux)\.*'
modelopts="--load-model-weights $modelpath --exclude-model-weights $exclude --freeze-model-weights $freeze \
--num-epochs 20 \
-o freeze_mode True -o fc_params [(512,0.1),(512,0.1)] \
 "
#-o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 

trainvalopts="--run-mode train,val "

source scripts/pheno2/train_Sophon_finetune_bbdeltaR.sh dryrun 1 --batch-size 1024 --start-lr 1e-3 $modelopts $trainvalopts 
1



#xww with Sophon-VQVAE
<<1
PREFIX=xww_Sophon-VQVAE_official_beta0p25_logits_1_vq_1_180_179_512_0.1_512_0.1_1024_5e-4_1
modelpath="/afs/ihep.ac.cn/users/k/kouzx/mlgpu/weaver-core/weaver/model/Sophon-VQVAE_official_beta0p25_logits_1_vq_1_180/net_epoch-179_state.pt"
test_epoch="model/${PREFIX}/net_epoch-9_state.pt"

DATADIR=/publicfs/cms/user/licq/datasets/JetClassII
DATADIR_HWW=/publicfs/cms/user/licq/condor_output/tagger_ext

config=./data_pheno/JetClassII_v2/JetClassII_full_nonscale_finetune_xww.yaml
freeze='(mod|module\.mod)\.(?!(cls_token|fc)).*'
fclayermatch='(?!(mod|module\.mod)\.(?!(cls_token|fc)).*)'
exclude='(((gen|module\.gen)|(mod|module\.mod)\.(fc)|(mod_.*|module\.mod_.*)\.(fc)).*)|(mod|module\.mod)\.(cls_token)|(mod|module\.mod)\.(cls_blocks_aux)\.*'

modelopts="--load-model-weights $modelpath --exclude-model-weights $exclude --freeze-model-weights $freeze \
-o freeze_mode True -o fc_params [(512,0.1),(512,0.1)] "
#-o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16
testopts="--log-file logs/${PREFIX}/test.log --model-prefix $test_epoch --data-split-num 1 -o label_cls_nodes ['label_0','label_1','label_2','label_3','label_4'] "

trainvalopts="--run-mode train,val --num-epochs 20 "
valextopts="-o eval_kw {'roc_kw':{'comp_list':[('XWW_rat0p6432','XWW_rat0p2'),('XWW_rat0p8','XWW_rat0p2'),('XWW_rat0p4','XWW_rat0p2'),('XWW_rat0p6432','XWW_rat0p4'),('XWW_rat0p8','XWW_rat0p4'),('XWW_rat0p8','XWW_rat0p6432'),('XWW_rat0p2','QCD'),('XWW_rat0p4','QCD'),('XWW_rat0p6432','QCD'),('XWW_rat0p8','QCD')],'label_inds_map':{'XWW_rat0p2':[0],'XWW_rat0p4':[1],'XWW_rat0p6432':[2],'XWW_rat0p8':[3],'QCD':[4]}}} "

source scripts/pheno2/train_Sophon_finetune_xww.sh run 3 --batch-size 1024 --start-lr 5e-4 $modelopts $trainvalopts $valextopts

#// ====== test mode =======
# //  * when using test mode, remember to additionally set: --data-split-num 1 -o label_cls_nodes ['XWW_rat0p2','XWW_rat0p4','XWW_rat0p6432','XWW_rat0p8','QCD'] 

#testopts="--run-mode test --log-file logs/${PREFIX}/test.log --data-split-num 1 -o label_cls_nodes ['XWW_rat0p2','XWW_rat0p4','XWW_rat0p6432','XWW_rat0p8','QCD'] "
#source scripts/pheno2/train_Sophon_finetune_xww.sh dryrun 0 --batch-size 1024 --start-lr 1e-4 $modelopts $testopts
1


# wcharge with Sophon-VQVAE
<<1
PREFIX=wcharge_Sophon-VQVAE_official_beta0p25_logits_1_vq_1_180_179_512_0.1_512_0.1_1024_5e-4_1
modelpath="/afs/ihep.ac.cn/users/k/kouzx/mlgpu/weaver-core/weaver/model/Sophon-VQVAE_official_beta0p25_logits_1_vq_1_180/net_epoch-179_state.pt"
test_epoch="model/${PREFIX}/net_epoch-19_state.pt"

DATADIR=/publicfs/cms/user/licq/datasets/JetClassII # ihep
DATADIR_X2P=/publicfs/cms/user/licq/condor_output/tagger_ext # ihep

config=./data_pheno/JetClassII_v2/JetClassII_full_nonscale_finetune_wcharge.yaml
freeze='(mod|module\.mod)\.(?!(cls_token|fc)).*'
fclayermatch='(?!(mod|module\.mod)\.(?!fc).*)'
exclude='(((gen|module\.gen)|(mod|module\.mod)\.(fc)|(mod_.*|module\.mod_.*)\.(fc)).*)|(mod|module\.mod)\.(cls_token)|(mod|module\.mod)\.(cls_blocks_aux)\.*'
modelopts="--load-model-weights $modelpath --exclude-model-weights $exclude --freeze-model-weights $freeze \
-o freeze_mode True -o fc_params [(512,0.1),(512,0.1)] \
 "
#-o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16

trainvalopts="--run-mode train,val --num-epochs 20 --data-split-num 50 "
valextopts="-o eval_kw {'roc_kw':{'comp_list':[('Xplus','QCD'),('Xplus','Xminus'),('Xminus','QCD')],'label_inds_map':{'Xplus':[0],'Xminus':[1],'QCD':[2]}}} "

source scripts/pheno2/train_Sophon_finetune_wcharge.sh run 2 --batch-size 1024 --start-lr 5e-4 $modelopts $trainvalopts $valextopts 
1