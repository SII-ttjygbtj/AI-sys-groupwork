# 使用 SkyPilot 进行开发
## 安装/登陆
官方文档：[https://docs.skypilot.co/en/latest/getting-started/installation.html](https://docs.skypilot.co/en/latest/getting-started/installation.html)

```bash
pip install skypilot
sky api login
```

执行 `sky api login` 后需输入以下 API 服务器端点：
```
http://skypilot:sglangjax20251018@34.162.107.179
```

## 示例 Yaml
```yaml
resources:
  accelerators: tpu-v6e-4 # 加速器类型
  accelerator_args:
    tpu_vm: True
    runtime_version: v2-alpha-tpuv6e # 可选
file_mounts:
  ~/.ssh/id_rsa: ~/.ssh/id_rsa # 挂载本地 SSH 密钥
setup: |
  chmod 600 ~/.ssh/id_rsa # 设置密钥权限
  rm ~/.ssh/config # 删除原有配置
  # 克隆代码仓库（跳过主机密钥验证）
  GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone git@github.com:Furion-cn/sgl-jax.git
  # 可选：安装依赖（取消注释启用）
  # cd /home/gcpuser/sky_workdir/sgl-jax && uv pip install -e python/

# 运行命令（取消注释启用）
# run: |
#   SGL_JAX_USE_JIT=1 JAX_COMPILATION_CACHE_DIR=/tmp/jax_compilation_cache 
#   uv run python -u -m sgl_jax.launch_server \
#     --model-path Qwen/Qwen-7B \
#     --engine-type jax \
#     --trust-remote-code \
#     --skip-server-warmup \
#     --dist-init-addr=0.0.0.0:10011 \
#     --nnodes=1 \
#     --tp-size=4 \
#     --device=tpu \
#     --random-seed=3 \
#     --node-rank=0 \
#     --mem-fraction-static=0.1 \
#     --max-prefill-tokens=4096 \
#     --download-dir=/tmp/ \
#     --kv-cache-dtype=bf16
```

## 核心操作场景
### 1. 启动开发机
#### 关键参数说明
- `--use-spot`：使用抢占式实例（成本更低）
- `-i 15`：空闲 15 分钟后自动停止（必填，可自定义时长）
- `--down`：停止时直接销毁服务器（释放资源）

#### 注意事项
- 创建过程中若报错，可先通过 `sky queue` 查看状态（可能已创建成功）
- 通过 SSH 登陆后，开发机不会进入空闲状态，也不会自动关停

#### 启动命令
```bash
sky launch test.yaml -y --use-spot --infra=gcp -i 5 --down
```

#### 启动日志示例
```
YAML to run: test.yaml
Running on cluster: sky-6f63-xl
Uploading files to API server
✓ Files uploaded View logs: ~/sky_logs/file_uploads/sky-2025-08-05-18-09-53-882143-c194e4ef.log
Launching a spot job that does not automatically recover from preemptions. To get automatic recovery, use managed job instead: sky jobs launch or sky.jobs.launch().
Considered resources (1 node):
INSTANCE          INFRA       VCPUs  Mem(GB)  GPUS       COST ($)  CHOSEN
GCP (us-central1-b) TPU-VM[Spot]  tpu-v6e-4:1          2.26       ✔️
Launching on GCP us-central1 (us-central1-b).
⠏ Launching View logs: sky api logs -l sky-2025-08-05-10-09-55-434972/provision.log
```

### 2. 查看开发机状态
查看当前用户的开发机/任务队列情况：
```bash
sky queue
```

#### 查看日志示例
```
Fetching and parsing job queue...
Fetching job queue for: sky-6f63-xl

Job queue of current user on cluster sky-6f63-xl
ID  NAME  USER  SUBMITTED  STARTED  DURATION  RESOURCES  STATUS  LOG  GIT COMMIT
1   xl          1 min ago  1 min ago  < 1s      1x[CPU:1+]  SUCCEEDED  ~/sky_logs/sky-2025-08-05-10-09-55-434972
```

### 3. 登陆开发机
```bash
ssh ${your_cluster_name}
```
> 替换 `${your_cluster_name}` 为实际集群名称（如 `sky-6f63-xl`）

### 4. 同步本地代码
将本地 `python` 目录同步到开发机指定路径：
```bash
rsync -Pavz python ${your_cluster_name}:/home/gcpuser/sky_workdir/sgl-jax/python
```

### 5. 启动测试集群
无需修改启动命令，仅需调整 Yaml 文件中的 `resources` 规格即可。

### 6. 提交测试任务
通过 `exec` 命令在所有节点上执行 `job.yaml` 中 `run` 部分的代码：
```bash
sky exec ${your_cluster_name} job.yaml
```

### 7. 停止测试任务
```bash
sky cancel ${your_cluster_name}
```

### 8. 删除服务器
```bash
sky down ${your_cluster_name}
```