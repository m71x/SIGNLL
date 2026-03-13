#!/bin/bash
set -e
export PROJECT_ID=early-exit-transformer-network
export TPU_NAME=node-5
export ZONE=us-central2-b

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/google_compute_engine

SSH_CMD="gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID}"

SETUP="pkill -f '[p]ython.*regret_training' || true; tmux kill-session -t regret 2>/dev/null || true; sleep 2; cd ~/SIGNLL && git fetch origin && git reset --hard origin/main"
LAUNCH='cd ~/SIGNLL && source ~/edel_env/bin/activate && rm -f ~/regret_training.log && tmux new -d -s regret "PJRT_DEVICE=TPU python3 src/llm_research/regret_training.py 2>&1 | tee ~/regret_training.log"'

for w in $(seq 0 7); do
    echo "=== Worker $w: setup ==="
    $SSH_CMD --worker=$w --command="$SETUP" 2>&1 | tail -2
    echo "=== Worker $w: launch ==="
    $SSH_CMD --worker=$w --command="$LAUNCH" 2>&1 | tail -2
    echo ""
done

echo "All workers launched. Checking in 30s..."
sleep 30
$SSH_CMD --worker=0 --command="head -5 ~/regret_training.log 2>/dev/null"
