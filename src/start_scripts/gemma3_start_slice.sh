export PROJECT_ID=early-exit-transformer-network
export TPU_NAME=node-5
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-64
export RUNTIME_VERSION=tpu-ubuntu2204-base

gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --version=${RUNTIME_VERSION}

#install compatible versions of torch and torch-xla 
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html"

#clone project repo onto all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="git clone https://github.com/m71x/SIGNLL"

#run inference
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="tmux new -d -s signll 'cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/tpu_job/main2.py'"

#check last system logs
tail -n 100 /var/log/syslog

#live logs
tail -f /var/log/syslog

#check real time running processes and resources
htop

#shows real time system performance based on memory, swaps, I/O, and CPU usage
vmstat 1

#crash context
journalctl -xe

# check disk free
df -h

#check open ports
ss -tulpn

#update workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="sudo apt update && sudo apt upgrade -y"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="sudo apt clean"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="sudo journalctl --vacuum-time=3d"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="sudo journalctl --vacuum-size=500M"

#flush hugging face cache
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="rm -rf ~/.cache/huggingface/*"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="sudo mount -o remount,size=150G /dev/shm"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="rm -rf /home/mikexi/siebert_model"

#run non-tmux job
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/tpu_job/npz_file_validation.py"

#run non-tmux job
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/training_job/train4.py"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="source edel_env/bin/activate && pip install torch"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv" && \
            python3.11 -m venv edel_env && \
            pip install --upgrade pip && \
            pip install easydel==0.2.0.2 \"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="source edel_env/bin/activate && pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip uninstall -y torch"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install easydel"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install 'accelerate>=0.26.0'"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install gcsfs"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="source edel_env/bin/activate && pip install -U bitsandbytes"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install --upgrade pip && pip install tensorflow tensorflow-datasets && pip install --upgrade easydel" 

#kill on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pkill -f -u mikexi python3"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="source edel_env/bin/activat pip install ml_dtypes"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install --upgrade easydel==0.2.0.2"

#run tmux training only
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="tmux new -d -s signll_train 'cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/llm_research/qwen_test.py'"

#pull changes from git repo
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="cd ~/SIGNLL && git fetch origin && git reset --hard origin/main"
  
#run eval (no tmux)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="cd ~/SIGNLL && source ~/edel_env/bin/activate && PJRT_DEVICE=TPU python3 src/llm_research/elarge_test.py"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=4 \
  --command="tmux new -d -s hf_upload 'cd ~/SIGNLL && PJRT_DEVICE=TPU python3 src/training_job/upload_to_hf.py'"

#clone xla onto all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} 
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="git clone https://github.com/pytorch/xla.git"


#unmount and remount to erase tmpfs
sudo umount /dev/shm
sudo mount -t tmpfs tmpfs /dev/shm

#install transformers on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="source edel_env/bin/activate && pip install transformers"

#install huggingface datasets library on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="pip install --upgrade datasets"

#upload script to all workers
gcloud compute tpus tpu-vm scp /home/mikexi/projects/signll_code/src/tpu_job/tpu_core_count.py ${TPU_NAME}:~ \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all

#https://cloud.google.com/tpu/docs/pytorch-pods - gcp documentation says use 2.5.0, but use 2.6.0 because there was a vulnerability in torch.load
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="pip install torch~=2.6.0 torch_xla[tpu]~=2.6.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html"


  #kill all user run processes: pkill -f -u mikexi python3

#install fasttext on all tpu workers (DO NOT USE ANYMORE)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip uninstall -y fasttext"

#install requests on all tpu workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install requests"

#install protobuf compiler on all TPUs (if you really need it, often it is default loaded)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="sudo apt-get update && sudo apt-get install -y protobuf-compiler"

#install gcld3
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install gcld3"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install fjformer"

gsutil cp ~/SIGNLL/final_model_stage2_gated.pt gs://encoder-models-2/result-models/

ls -laR /home/mikexi/siebert_model | less

#check top 20 largest directories
sudo du -ahx / | sort -rh | head -n 20


gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="sudo truncate -s 0 /var/log/syslog && sudo truncate -s 0 /var/log/kern.log"
# Run this on the worker(s) with full disks

#ssh into worker 2
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=3