# Instructions for GCP cloud VMs with multiple GPUs


```
sudo nvidia-smi -pm 1
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalytics
cd DeepVideoAnalytics/deploy/gpu
docker-compose -f docker-compose-multi-gpu.yml up -d
```