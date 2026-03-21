#!/bin/bash
set -e
export PROJECT_ID=early-exit-transformer-network
export TPU_NAME=node-5
export ZONE=us-central2-b

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/google_compute_engine

SSH_CMD="gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID}"

# Phase 1: Kill all workers first (cleanup)
CLEANUP='tmux kill-session -t regret 2>/dev/null || true; pkill -9 -f "regret_training|easydel|worker_main" || true; sleep 2; rm -f /tmp/easydel_*.sock /tmp/libtpu_lockfile'

echo "=== Phase 1: Cleanup all workers ==="
for w in $(seq 0 7); do
    echo "  Cleaning worker $w..."
    $SSH_CMD --worker=$w --command="$CLEANUP" 2>&1 | tail -2 || true
done

echo "Waiting 5s for TPU devices to release..."
sleep 5

# Phase 2: Git sync all workers
SYNC="cd ~/SIGNLL && git fetch origin && git reset --hard origin/main"
echo "=== Phase 2: Git sync all workers ==="
for w in $(seq 0 7); do
    echo "  Syncing worker $w..."
    $SSH_CMD --worker=$w --command="$SYNC" 2>&1 | tail -2
done

# Phase 3: Launch all workers as fast as possible
LAUNCH='cd ~/SIGNLL && source ~/edel_env/bin/activate && rm -f ~/regret_training.log && tmux new -d -s regret "PJRT_DEVICE=TPU python3 -u src/llm_research/regret_training.py 2>&1 | tee ~/regret_training.log"'

echo "=== Phase 3: Launch all workers ==="
for w in $(seq 0 7); do
    echo "  Launching worker $w..."
    $SSH_CMD --worker=$w --command="$LAUNCH" 2>&1 | tail -2
done

echo "All workers launched. Checking in 60s..."
sleep 60
$SSH_CMD --worker=0 --command="tail -10 ~/regret_training.log 2>/dev/null"
