# GPU 机器重新分配后的状态恢复

机器重新分配后，`/root` 下的东西（仓库、`.bashrc` 里的 `PYTHONPATH`、手动替换过的 `.so`）
都可能没了，镜像自带的是**不含 KVPilot 改动**的组件。按下面三步恢复，每步做完都有自查。

## 0. 连上新机器

重新分配后 IP / 端口通常会变，先改本机 `~/.ssh/config` 里 `Host Taiji-GPU` 的
`HostName` / `Port`，然后：

```bash
ssh -o ConnectTimeout=15 Taiji-GPU 'nvidia-smi -L | wc -l; ls /cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507 | head -3'
```

应看到 8 张卡，模型目录在共享存储上（`dynamo_script.sh` / `disagg_lmcache.sh` 写死了这个路径）。

## 1. 同步代码到 /root/kvpilot

```bash
bash ~/Desktop/code/utils/sync_to_devcloud.sh local:$HOME/Desktop/code/kvpilot Taiji-GPU:/root
```

- 新机器上 `/root/kvpilot` 不存在，全是新增，不用先 `-n`。脚本默认不删除目标端多余文件，
  要清理时加 `--delete`（先 `-n --delete` 看影响）。
- 同步跳过 `.git/`、`build/`、`*.so`、`__pycache__/` 等，见脚本里的 `EXCLUDES`。
- `Mooncake/extern/{pybind11,yalantinglibs}` 本地没 init，传过去是空目录；编译时用镜像里
  的 `/opt/pybind11` 补（见 2.3）。
- 自查：`ssh Taiji-GPU 'ls /root/kvpilot; du -sh /root/kvpilot'`，应有
  `vllm-dev lmcache-dev Mooncake dynamo-dev datasets paper-figure utils docs`。

### 模型路径：直接用 CFS

**默认用 CFS，不要先拷本地盘。** `dynamo_script.sh` 默认已是：

```bash
export MODEL=/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507
```

CFS 冷启动会比本地盘慢（权重按需读），但通常仍快过等 57G 拷完再起；2026-11-04 实测
双 1P1D 从 CFS 起服约 7 分钟就绪。只有确认 CFS 读极慢导致 setup 超时（旧记录曾到
600 s）时，才考虑拷到 `/root/models/Qwen/...` 再 `export MODEL=` 指过去。

2026-11-11 重新分配后就遇到了：vLLM 预取权重只有 15–40 MB/s，10 分钟才读完 1/8，setup 在 900 s
（`SETUP_MODELS_WAIT_S`）超时。16 个分片并行 `cp` 约 10 分钟拷完，之后每次起服务约 3 分钟：

```bash
ssh Taiji-GPU 'S=/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507; D=/root/models/Qwen/Qwen3-30B-A3B-Instruct-2507
mkdir -p $D && cd $S && nohup setsid bash -c "ls | xargs -P 16 -I{} cp {} $D/{}; echo COPY_EXIT \$?" > /root/models/copy.log 2>&1 < /dev/null &'
```

`run_ab.sh`、`run_one.sh`、`dynamo_script.sh` 都读 `MODEL` 环境变量，起服务前 `export MODEL=$D` 即可。

### 补 PYTHONPATH

仓库的 Python 代码不是 pip 安装的，**只靠 `PYTHONPATH` 生效**。新机器的 `.bashrc` 里没有它：

```bash
ssh Taiji-GPU 'grep -q kvpilot/vllm-dev /root/.bashrc || cat >> /root/.bashrc <<"EOF"
export PYTHONPATH=/root/kvpilot/vllm-dev:/root/kvpilot/lmcache-dev:/root/kvpilot/dynamo-dev/components/src:/root/kvpilot/dynamo-dev/lib/bindings/kvbm/python
EOF'
```

`.bashrc` 只对交互 shell 生效，非交互 `ssh` 里仍要自己 export，见
[benchmark.md](benchmark.md)「非交互 ssh 必须自己 export PYTHONPATH」。

## 2. 判断要不要编译，并编译

四个 fork 里，**vLLM 的改动全是 Python，不用编译**；dynamo 的 `components/` 也是 Python。
要编译的只有下面三样，判断依据都是「装着的二进制里有没有 KVPilot 才有的符号/字符串」：

| 组件 | 改了什么 | 装在哪 | KVPilot 标记 |
|---|---|---|---|
| Mooncake | `mooncake-store/` + `mooncake-integration/store/store_py.cpp` | `mooncake.store` 的 `.so`、`/usr/local/bin/mooncake_master` | `cuda_host_register_segment_succeeded`、`preload_shared_memory_segments`、`GetSharedMemorySegments`；SSD 加载还要 Python 属性 `batch_locate_local_ssd`；SSD 命中 load-through 还要 Python 属性 `batch_put_start_local`（client 把 put 拆成三个调用）；placement 还要 Python 属性 `batch_set_group_placement`、`batch_remove_local_disk`、`batch_offload_local`（exclusive 直写 SSD），以及 master 里的 `KVPilot placement=` 字符串（2026-11-19 重构把 `SetGroupPriority` RPC 换成了 `SetGroupPlacement`，三个策略都按组淘汰，`store` 与 `mooncake_master` 必须一起换，否则 RPC 对不上） |
| dynamo Rust | `lib/llm/src/preprocessor.rs`、`protocols/openai/validate.rs` | `dynamo/_core.abi3.so` | `cache_use_control`、`cache_use_max_tokens`、`forced_output_tokens`；`local SSD`（`cache_use_control` 允许 4）；`one entry per`（`cache_use_max_tokens` 允许按层级的列表） |
| LMCache `c_ops` | `csrc/{mem_kernels.cu,mem_kernels.cuh,ssd_loader.cu,ssd_loader.cuh,pybind.cpp}`（未提交） | `dist-packages/lmcache/c_ops*.so` | Python 属性 `LayerwiseLoader`、`SsdLayerwiseLoader`；SSD 读带宽上限还要 `.so` 里有字符串 `KVPILOT_SSD_READ_GIBPS`；HBM→DRAM 写（store、exclusive）还要 `batched_memcpy_d2h`；SSD 命中 load-through 还要 `SsdLayerwiseLoader.wait_writeback` / `set_writeback`；`start()` 总带着写回参数，`c_ops` 缺它们时每次 SSD 加载都报错，exclusive 下也一样 |

### 2.1 一次查完

```bash
ssh Taiji-GPU 'bash -s' <<'EOF'
export PYTHONPATH=/root/kvpilot/vllm-dev:/root/kvpilot/lmcache-dev:/root/kvpilot/dynamo-dev/components/src:/root/kvpilot/dynamo-dev/lib/bindings/kvbm/python
cnt() { [ -f "$2" ] && grep -c "$1" "$2" || echo missing; }
ST=$(python3 -c 'import mooncake.store as s; print(s.__file__)' 2>/dev/null)
MM=$(command -v mooncake_master)
CORE=$(python3 -c 'import dynamo._core as c; print(c.__file__)' 2>/dev/null)
echo "mooncake store  $ST"
echo "  cuda_host_register_segment_succeeded: $(cnt cuda_host_register_segment_succeeded "$ST")"
echo "  preload_shared_memory_segments:       $(cnt preload_shared_memory_segments "$ST")"
echo "mooncake_master $MM"
echo "  GetSharedMemorySegments:              $(cnt GetSharedMemorySegments "$MM")"
echo "  KVPilot placement=:                   $(cnt 'KVPilot placement=' "$MM")"
echo "dynamo _core    $CORE"
for s in cache_use_control cache_use_max_tokens forced_output_tokens "local SSD" "one entry per"; do
  echo "  $s: $(cnt "$s" "$CORE")"; done
python3 -c 'import mooncake.store as s; S = s.MooncakeDistributedStore; print(" ", *(f"{a}: {hasattr(S, a)}" for a in ("batch_locate_local_ssd", "batch_put_start_local", "batch_set_group_placement", "batch_remove_local_disk", "batch_offload_local")))'
python3 -c 'import lmcache.c_ops as c; print("lmcache c_ops  ", c.__file__); print("  LayerwiseLoader:", hasattr(c, "LayerwiseLoader"), " SsdLayerwiseLoader:", hasattr(c, "SsdLayerwiseLoader"), " wait_writeback:", hasattr(getattr(c, "SsdLayerwiseLoader", None), "wait_writeback"), " batched_memcpy_d2h:", hasattr(c, "batched_memcpy_d2h"))' 2>&1 | grep -v INFO
CO=$(python3 -c 'import lmcache.c_ops as c; print(c.__file__)' 2>/dev/null)
echo "  KVPILOT_SSD_READ_GIBPS: $(cnt KVPILOT_SSD_READ_GIBPS "$CO")"
python3 -c 'import vllm, lmcache; print("vllm   ", vllm.__file__); print("lmcache", lmcache.__file__)' 2>&1 | grep -v INFO
EOF
```

- 任何一个计数是 0 或 `missing`、属性是 `False`，对应组件就要重编。
- 最后两行必须指向 `/root/kvpilot/...`；指向 `dist-packages` 说明 `PYTHONPATH` 没生效。
- `cuda_host_register_segment_succeeded` 只在 Mooncake 的 **CUDA 构建**里有。非 CUDA 构建
  只有 `not_a_cuda_build`，此时 shm segment（固定开）会在注册 segment 时直接报错。

下面三节互不依赖，哪个缺就编哪个。编译都在 `/root/kvpilot` 下对应的仓库里做，中间产物和日志
留在仓库里，下次可以增量编译。每节安装前都先备份现有产物，出问题可以直接换回去。

和同步脚本的关系：
- **`build/` 目录**：在同步脚本的排除列表里，带不带 `--delete` 都不会被删，LMCache 和
  Mooncake 的产物就放在这里。
- **dynamo 的 `target/`**：不在排除列表里。平时同步不加 `--delete`，所以不受影响；一旦加了
  `--delete` 做全量同步，它会被删掉，下次 dynamo 只能从头编。

### 2.2 LMCache c_ops（约 3 分钟）

在 `lmcache-dev/build/` 下只编 `c_ops` 这一个扩展，不跑 `setup.py` 的完整安装。源文件列表和
编译参数跟 `lmcache-dev/setup.py` 的 `cuda_extension()` 一致：

```bash
ssh Taiji-GPU 'bash -s' <<'EOF'
B=/root/kvpilot/lmcache-dev/build && mkdir -p $B
cat > $B/build_cops.py <<"PY"
import os, sys
from setuptools import setup
from torch.utils import cpp_extension
root = sys.argv.pop(1)
srcs = [os.path.join(root, "csrc", f) for f in (
    "pybind.cpp", "mem_kernels.cu", "ssd_loader.cu", "mp_mem_kernels.cu", "cal_cdf.cu", "ac_enc.cu", "ac_dec.cu",
    "pos_kernels.cu", "mem_alloc.cpp", "utils.cpp", "event_recorder.cpp", "completion_recorder.cpp")]
abi = "-D_GLIBCXX_USE_CXX11_ABI=1"
setup(name="cops_only", version="0",
      ext_modules=[cpp_extension.CUDAExtension("lmcache.c_ops", sources=srcs,
                   extra_compile_args={"cxx": [abi, "-std=c++17"], "nvcc": [abi]})],
      cmdclass={"build_ext": cpp_extension.BuildExtension})
PY
INC=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
       TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 CPATH=$INC CPLUS_INCLUDE_PATH=$INC
cd $B && nohup bash -c "python3 build_cops.py /root/kvpilot/lmcache-dev build_ext --build-lib $B/cops_out --build-temp $B/cops_tmp; echo EXIT \$?" > $B/cops-build.log 2>&1 &
EOF
ssh Taiji-GPU 'grep ^EXIT /root/kvpilot/lmcache-dev/build/cops-build.log'   # 等到出现 EXIT 0
```

**不要在仓库根下运行这个脚本**，要在 `build/` 里跑，源文件用绝对路径传进去。原因是
`lmcache-dev/pyproject.toml` 配了 setuptools-scm：在仓库根下跑时，setuptools 会去读它；同步
过去的代码又没有 `.git`，结果报 `setuptools-scm was unable to detect version` 后直接退出。
输出都写在 `build/` 下，`cops_tmp` 里留着 `.o`，再编时只重编改过的文件。

`CPATH` 不能省：`/usr/local/cuda-13.0` 里没有 `cusparse.h`，torch 头文件要用，只有 pip 的
`nvidia/cu13/include` 里有。省掉会报 `fatal error: cusparse.h: No such file or directory`。

`EXIT 0` 后安装（`cp -n` 保证只备份镜像原版，不会被后续编出来的版本覆盖）：

```bash
ssh Taiji-GPU 'D=/usr/local/lib/python3.12/dist-packages/lmcache; F=c_ops.cpython-312-x86_64-linux-gnu.so
cp -n $D/$F /root/c_ops.so.orig && cp /root/kvpilot/lmcache-dev/build/cops_out/lmcache/$F $D/ && ls -la /root/c_ops.so.orig $D/$F'
```

### 2.3 Mooncake（约 3 分钟，换 `store` 模块和 `mooncake_master`）

镜像里的 Mooncake 来自 `dynamo-dev/deploy/taiji-cluster/Dockerfile`，构建时用的源码不一定
带 KVPilot 改动，**以 2.1 的检查结果为准**。镜像自带的 Mooncake（9/3 构建）有 shm HBM 的改动，
但没有 SSD 加载要的几处：`batch_locate_local_ssd`、bucket 写盘后的 `fdatasync`、磁盘副本里的
文件位置（`LocalDiskDescriptor.file_path / value_offset`，跨 worker 读 SSD 用）。最后这处改了
client 和 master 之间的消息格式，**`store` 和 `mooncake_master` 必须一起换**，只换一边会解析错。
在 `Mooncake/build/` 下编：

```bash
ssh Taiji-GPU 'bash -s' <<'EOF'
cd /root/kvpilot/Mooncake
[ -f extern/pybind11/CMakeLists.txt ] || cp -r /opt/pybind11/. extern/pybind11/
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDAToolkit_ROOT=/usr/local/cuda-13.0
mkdir -p build && cd build
cmake .. > cmake.log 2>&1; echo "CMAKE $?"
grep -c "mooncake_store: CUDAToolkit detected" cmake.log
EOF
```

- 最后一行**必须是 1**。是 0 说明 CMake 没找到 CUDA，编出来的是非 CUDA 版，shm HBM 用不了，
  不要继续往下装。`USE_CUDA` 不用打开，`mooncake_store` 会自己探测 CUDAToolkit。
- `extern/pybind11` 在本地是没 init 的空 submodule，第一行把镜像里的 `/opt/pybind11` 复制
  进去。同步脚本把未 init 的 submodule 当成保护路径，即使加 `--delete` 也不会把复制进去的
  文件删掉。

确认是 1 后，只编 `store` 和 `mooncake_master` 两个目标并替换（2026-11-03 实测）：

```bash
ssh Taiji-GPU 'bash -s' <<'EOF'
cd /root/kvpilot/Mooncake/build && export PATH=/usr/local/cuda-13.0/bin:$PATH
nohup bash -c 'make -j96 store mooncake_master; echo "EXIT $?"' > build.log 2>&1 < /dev/null &
EOF
ssh Taiji-GPU 'tail -1 /root/kvpilot/Mooncake/build/build.log'   # 等到 EXIT 0
ssh Taiji-GPU 'M=/usr/local/lib/python3.12/dist-packages/mooncake; S=store.cpython-312-x86_64-linux-gnu.so; B=/root/kvpilot/Mooncake/build
mkdir -p /root/mooncake.orig && cp -n $M/$S /usr/local/bin/mooncake_master /root/mooncake.orig/
cp $B/mooncake-integration/$S $M/ && cp $B/mooncake-store/src/mooncake_master /usr/local/bin/'
```

自查：`python3 -c 'import mooncake.store as s; print(hasattr(s.ReplicaDescriptor, "get_local_disk_descriptor"))'`
应为 True。

`import mooncake.store` 加载的是带 ABI 标签的 `store.cpython-312-x86_64-linux-gnu.so`，同目录下
6 月的 `store.so` 是 pip 包原带的，不会被用到。

改了 master 侧代码时才需要完整编译并安装。安装位置：`mooncake_master` 到 `/usr/local/bin`，
`store` 等 Python 模块到 `mooncake` 包目录（未实测）：

```bash
ssh Taiji-GPU 'bash -s' <<'EOF'
ST=$(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))')
mkdir -p /root/mooncake.orig && cp -n $ST/*.so /usr/local/bin/mooncake_master /root/mooncake.orig/
cd /root/kvpilot/Mooncake/build
export PATH=/usr/local/cuda-13.0/bin:$PATH
nohup bash -c 'make -j32 && make install && ldconfig; echo "EXIT $?"' > build.log 2>&1 &
EOF
# 等结果
ssh Taiji-GPU 'tail -1 /root/kvpilot/Mooncake/build/build.log'
```

装完重跑 2.1，三个 Mooncake 标记都应 ≥ 1。`make install` 装的目录一定要是 `import mooncake`
实际加载的那个目录。如果镜像里另有一份 pip 装的 `mooncake-transfer-engine`，而 import 走的是
那一份，那么 2.1 里的计数还会是 0。

### 2.4 dynamo Rust（约 6 分钟）

只有改了 `lib/` 下的 Rust 才需要编。完整说明（首次搭工具链、常见坑、怎么验证改动真的编进去了）
在飞书 `SnUXdzCJvoyczZxWTCec08nRnBc` 的「dynamo 编译」一节。新机器上照下面的步骤做就行：

1. 镜像里的 crates 已经缓存好，可以离线编译，但 `utoipa-swagger-ui` 的构建脚本还要下载 swagger
   的 zip。GPU 机器上不了 GitHub，所以先从本机传上去：

   ```bash
   mkdir -p /tmp/swg && curl -sSL -o /tmp/swg/swagger.zip \
       https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.17.14.zip
   base64 < /tmp/swg/swagger.zip | tr -d '\n' | \
       ssh Taiji-GPU 'base64 -d > /root/swagger-ui-v5.17.14.zip && sha256sum /root/swagger-ui-v5.17.14.zip | cut -c1-16'
   ```

   输出应为 `481244d0812097b1`。不能写成 `ssh ... "echo '$B64' ..."`，zip 有 4 MB，会报
   `argument list too long`。

2. 编 wheel。在 `dynamo-dev/lib/bindings/python` 下编，这个 crate 自成一个 workspace，中间
   产物在它下面的 `target/`，wheel 在 `target/wheels/`，和 10/30 那次的做法一样。平时同步不带
   `--delete`，`target/` 会一直留着，可以增量编译；用 `--delete` 全量同步后要从头编。

   ```bash
   ssh Taiji-GPU 'bash -s' <<'EOF'
   P=/root/kvpilot/dynamo-dev/lib/bindings/python
   export PATH=/usr/local/cargo/bin:$PATH RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
   export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=32
   export SWAGGER_UI_DOWNLOAD_URL=file:///root/swagger-ui-v5.17.14.zip
   ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
   cd $P && rm -rf target/wheels
   nohup bash -c 'maturin build --release --offline; echo BUILD_EXIT=$?' > build.log 2>&1 &
   EOF
   ssh Taiji-GPU 'grep BUILD_EXIT /root/kvpilot/dynamo-dev/lib/bindings/python/build.log'   # 等到出现 BUILD_EXIT=0
   ```

   - `ulimit -n` 不能省。默认值是 1024，`rustix` 的构建脚本会报 `could not execute process`。
   - 如果编完 2.1 的计数还是 0，先确认本机改动已经同步上去了：
     `grep -c cache_use_control /root/kvpilot/dynamo-dev/lib/llm/src/protocols/openai/validate.rs`。

3. 安装：

   ```bash
   ssh Taiji-GPU 'C=/usr/local/lib/python3.12/dist-packages/dynamo/_core.abi3.so
   cp -n $C /root/_core.abi3.so.orig
   pip install --force-reinstall --no-deps /root/kvpilot/dynamo-dev/lib/bindings/python/target/wheels/ai_dynamo_runtime-*.whl 2>&1 | tail -1
   python3 -c "import dynamo._core; print(\"import ok\")"'
   ```

## 3. 最后确认

1. 重跑 2.1，所有标记都应 ≥ 1 或 `True`。
2. 第一次起服务后，按 [benchmark.md](benchmark.md) 检查仓库代码有没有真正生效：

   ```bash
   grep -c 'KVPilot placement=' dynamo-log/<name>-logs/prefill-0.log   # 0 = 跑成了 stock vLLM
   ```

## 备份文件

上面每一步都只在第一次运行时备份镜像原版（`cp -n`），文件名带 `.orig`：

| 备份 | 原位置 |
|---|---|
| `/root/c_ops.so.orig` | `dist-packages/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so` |
| `/root/mooncake.orig/` | `mooncake` 包目录下的 `*.so`、`/usr/local/bin/mooncake_master` |
| `/root/_core.abi3.so.orig` | `dist-packages/dynamo/_core.abi3.so` |

需要回到镜像原版时，把备份文件拷回原位置即可。
