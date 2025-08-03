#!/bin/bash -x

RUN=$1
GPUS=$2

if [ -z $GPUS ]; then
    echo "Usage: $0 <ngpu>"
    exit 1
fi
NGPUS=$(echo $GPUS | tr "," "\n" | wc -l)

cmdlineopts="${@:3}"

current_dir=`pwd`
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

# prepare training dataset
#  XWW dataset, train:val:test = 64:16:16
trainset_xww=$(for n in 0p2 0p4 0p6432 0p8; do for i in $(seq 0 63); do echo -n "XWW_rat${n}:${DATADIR_HWW}/train_hyy4q_fixmassrat_${n}_ntuple/ntuples_$i.root "; done; done)
valset_xww=$(for n in 0p2 0p4 0p6432 0p8; do for i in $(seq 64 79); do echo -n "XWW_rat${n}:${DATADIR_HWW}/train_hyy4q_fixmassrat_${n}_ntuple/ntuples_$i.root "; done; done)
testset_xww=$(for n in 0p2 0p4 0p6432 0p8; do for i in $(seq 80 95); do echo -n "XWW_rat${n}:${DATADIR_HWW}/train_hyy4q_fixmassrat_${n}_ntuple/ntuples_$i.root "; done; done)

#  QCD dataset, train:val:test = 56:14:14 (1/5 of JetClassII dataset)
trainset_qcd=$(for i in $(seq -w 0000 0055); do echo -n "QCD:${DATADIR}/Pythia/QCD_$i.parquet "; done)
valset_qcd=$(for i in $(seq -w 0280 0293); do echo -n "QCD:${DATADIR}/Pythia/QCD_$i.parquet "; done)
testset_qcd=$(for i in $(seq -w 0350 0363); do echo -n "QCD:${DATADIR}/Pythia/QCD_$i.parquet "; done)

ARG="--network-config networks/pheno2/example_SophonSharedBody.py -o num_classes 5 -o fc_params [(512,0.1)] \
--use-amp --batch-size 512 --start-lr 5e-4 --samples-per-epoch $((5000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024 / $NGPUS)) --num-epochs 20 --optimizer ranger \
--num-workers 4 --fetch-step 1.0 --data-split-num 50 \
--data-train $trainset_xww $trainset_qcd \
--data-val $valset_xww $valset_qcd \
--data-test $testset_xww $testset_qcd \
--data-config $config \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "


if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ] || [ $RUN == "autorecover" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

if [ $GPUS == "cpu" ]; then
    cmd="python train.py $ARG $cmdlineopts "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
    cmd="python train.py --gpus $GPUS $ARG $cmdlineopts "
else
    # GPU list is separated by comma
    export CUDA_VISIBLE_DEVICES=$GPUS
    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS train.py --backend nccl $ARG $cmdlineopts "
fi

echo Run command: $cmd

if [ $RUN == "dryrun" ] || [ $RUN == "run" ]; then
    $cmd
elif [ $RUN == "autorecover" ]; then
    epochopts=""
    # if the training is halted, resume from the last epoch
    while true; do
        $cmd $epochopts
        ret=$?
        if [ $ret -eq 0 ]; then
            break
        fi
        echo "Error: return code $ret"
        # match model/${PREFIX}/net_epoch-(\d+)_state.pt and extract the maximum epoch number
        maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
        if [ -z $maxepoch ]; then
            epochopts=""
        else
            epochopts="--load-epoch $maxepoch"
            echo "Resuming from epoch $maxepoch"
        fi
        sleep 10
    done
fi
