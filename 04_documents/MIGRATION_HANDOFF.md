# 交接简报 — 给新主机上的 Claude Code

我正在把工作环境从旧机(Windows 11 + WSL2 Ubuntu 24.04)迁到这台新主机。
数据在一块梵想 1TB 移动硬盘上(exFAT,旧机上是 F 盘)。请按下面的状态接手。

完整手册:https://claude.ai/code/artifact/fe7ebcf7-7de3-450d-af26-117bbc97d246
(先 WebFetch 读这个链接,里面有全部命令和对照表)

## 最重要的一件事:硬盘状态未验证

上一次拷贝到归档区 90% 时被终止,盘随后从系统断开且没走安全弹出。
exFAT 无日志,有目录结构损坏的可能。**在验盘通过之前不要还原任何东西。**

验盘两步:
1. `chkdsk F:` 检查文件系统
2. 校验核心归档:
   - 文件 `F:\TRANSFER\To_Desktop\wsl\wsl-home-20260909-0829.tar.zst`
   - 大小必须是 6891300008 字节
   - SHA256 必须是 db0146b0c2def568fadb2346b357a264de0787f5811aaf9578770d2584bb7641

对不上就停,告诉我。旧机完好无损,回去重新打包即可。

## 掉线前最后一次扫描的数字(存疑项)

| 路径 | 预期 | 实测 | 判断 |
|---|---|---|---|
| TRANSFER\To_Desktop\wsl | 6.42 GB | 6.42 GB | 已验 SHA256 |
| TRANSFER\To_Desktop\windows | 15.0 GB | 14.99 GB | 一致 |
| TRANSFER\To_Desktop\gamesaves | 36.6 GB | 8.29 GB | **存疑** |
| TRANSFER\To_Desktop\config | <0.5 GB | 2.49 GB | 偏高,应是编辑器缓存 |
| BACKUP\Projects_Backup | 20+ GB | 4.93 GB | **存疑** |
| MEDIA\Personal\Non_Movie | 118 GB | 读不到 | 扫描时已掉线 |

存疑项可能只是扫描被打断的假象,需要重新实测确认。

## 已经确定安全的部分

- 六个仓库全部提交并推送到 GitHub(含新建的私有仓库 Danmaku)
- 旧机的 C、D、E 三块盘和 WSL 全部原封未动,是可靠退路
- 核心归档在掉线前已验过 SHA256,当时是完好的

## 磁盘方案:不分区(单块 Samsung 990 Pro 2TB)

**结论已从「分三个区」改为「整盘一个 C 卷」。** 理由:

1. NVMe 上分区零性能收益,分区提速是机械盘时代的经验。
2. 磨损均衡跨整盘工作不看分区,但**空闲空间按分区算** —— 某个卷填满时其他卷的空闲帮不上忙,
   控制器可支配余量变少,极端情况读写掉三到五成。单卷所有空闲共享。
3. 「重装保住数据」不需要分区,Windows 11 的「保留我的文件」不依赖独立数据卷。
   真正的保障是移动硬盘和 GitHub。
4. 尺寸一定会算错:WSL 的 ext4.vhdx 只涨不缩(2026 现状:fstrim 无用,
   sparse VHD 因损坏风险被标为不安全,diskpart 拒绝处理 sparse 文件)。

唯一站得住的反对意见是「失控写入撑爆系统盘」,但 2TB 里系统只占 164 GB,
需要约 1.7 TB 失控写入才会发生,用磁盘空间守卫比锁死 500 GB 划算。
**唯一会改口的情况:要装原生 Linux 双系统。**

### 目录结构

```
C:\
├─ Work\      研究:Projects Docs Courses Data Tools
├─ Media\     娱乐:Games SteamLibrary
├─ Cache\     全部可再生:pip conda huggingface npm cargo rustup models
└─ WSL\Ubuntu\   ext4.vhdx
```

Users 目录保持默认,**不重定向 Documents** —— 单卷之后没有空间理由,移动反增出错面。

### 四个防腐化机制(靠自觉一定失败)

```powershell
# 1. 所有缓存赶进 C:\Cache,可再生数据集中一处,想清整个删掉
[Environment]::SetEnvironmentVariable('PIP_CACHE_DIR','C:\Cache\pip','User')
[Environment]::SetEnvironmentVariable('CONDA_PKGS_DIRS','C:\Cache\conda','User')
[Environment]::SetEnvironmentVariable('HF_HOME','C:\Cache\huggingface','User')
[Environment]::SetEnvironmentVariable('CARGO_HOME','C:\Cache\cargo','User')
[Environment]::SetEnvironmentVariable('RUSTUP_HOME','C:\Cache\rustup','User')
npm config set cache C:\Cache\npm --global

# 2. WSL 虚拟磁盘移出 AppData,为了可见可压缩,不是为省空间
wsl --shutdown
wsl --manage Ubuntu --move C:\WSL\Ubuntu

# 3. 把 C:\Work 及子目录固定到快速访问并加进「库」,让它取代 Documents 成为默认导航目标

# 4. 磁盘空间守卫,代替分区硬边界(计划任务每天跑)
$free = (Get-Volume -DriveLetter C).SizeRemaining/1GB
if ($free -lt 150) { msg * "C 盘剩余 $([math]::Round($free)) GB,检查 C:\Cache 与 C:\WSL" }
```

### 脚本已相应修改

`11_restore_windows.ps1` 不再依赖 D 盘:参数从 `-DataDrive` 改为 `-WorkRoot`,默认 `C:\Work`。
脚本还会自动把 Zotero prefs.js 里的 dataDir 改写成新路径并备份原文件,不需要手工重设。

```powershell
powershell -ExecutionPolicy Bypass -File F:\scripts\11_restore_windows.ps1 -Drive F: -WorkRoot C:\Work
```

## 重建要点

- WSL 用户名沿用 `jain_farstrider`,否则归档里的绝对路径要额外处理
- 还原脚本在盘根 `F:\scripts\`,不需要先解包
- **conda 环境必须用 `conda-nlp-full.yml`**(28 conda + 223 pip)。
  `conda-nlp-history.yml` 只有 python 和 pip 两项,是空的,不能用。
- 目标版本:Python 3.11.15、Node v24.14.1、R 4.3.3、Ubuntu 24.04
- R 有 244 个包,apt 25 个手动安装包(关键是 r-base-core 和中文 LaTeX 那几个)
- SSH 私钥还原后必须 `chmod 600`,否则 git 推不动

## 三个已知陷阱

1. **活的 Zotero 库在 `C:\Work\Docs\Papers\Zotero`(还原后)**,不是用户目录下那个;还原后落在 `C:\Work\Docs\Papers\Zotero`。
   `TRANSFER\stale\Zotero-CDrive-DEAD` 是旧机 3 月的失效副本,不要还原。
2. **`D:\Projects` 是 3 至 4 月的过期快照**,WSL 侧才是权威,不要反向覆盖。
3. **Danmaku 的 .md 文档按仓库策略不入 Git**,只存在于归档包里。

## 遗留待办

- 归档区媒体主动放弃于 90%,源文件仍在旧机 D 盘 E 盘
- My-Web-V2 备份因路径写错被跳过,脚本已修
- `D:\下载\Compressed` 首轮有文件复制失败(exit 11),疑似文件名过长,未核对

## 用户名已更改 —— 还原后必须修路径

新机的用户名与旧机不同(旧机是 `jain_farstrider`)。影响如下:

**不受影响:**
- 归档内路径是相对的(`./projects/...`),解包到任何 home 都正确
- Windows 侧脚本用 `$env:USERPROFILE`,自动适配新的 Windows 用户名

**会断的:** 九个文件里有硬编码的 `/home/jain_farstrider`,其中两个 Python 脚本
的 `ROOT = "/home/jain_farstrider/..."` 会直接跑不动。

| 文件 | 处数 | 后果 |
|---|---|---|
| `~/.bashrc` | 4 | conda 初始化失败,`conda` 命令找不到 |
| `~/.gitconfig` | 2 | gh 凭证助手路径失效 |
| `03_outputs/.../analyze_arbitration.py` | 1 | ROOT 常量错误,脚本跑不动 |
| `03_outputs/.../compute_gold_vs_models.py` | 1 | 同上 |
| `Ukrainian-Poetry/scripts/bootstrap_public_repo.py` | 1 | 路径错误 |
| `.claude/settings.local.json` ×2 | — | 权限规则路径失效 |
| `migration/env/conda-*.yml` ×4 | 各 1 | prefix 行,conda 实际会忽略 |

修复脚本 `12_fix_paths.sh` 在盘根 `F:\scripts\`,默认干跑:

```bash
bash /mnt/f/scripts/12_fix_paths.sh                 # 先看会改什么
bash /mnt/f/scripts/12_fix_paths.sh --apply         # 确认后执行,原文件备份为 .bak-path
```

**它刻意不改日志与运行清单。** 全盘扫描能扫出 427 个含旧路径的文件,
但其中绝大多数是 `.log` 和 `input_manifest.json` 这类历史记录 ——
改写它们等于伪造过去的运行痕迹,对复现性有害。只有会被执行的文件才需要改。

如果新机的 WSL 用户名也可以自己定,**最省事的做法是仍然用 `jain_farstrider`**,
这样一处都不用改。

## 验证清单

还原完成后逐项跑通,全过再考虑清理旧机:

- `ssh -T git@github.com` 能认证
- 六个仓库 dirty=0,Taiwan 项目最新提交为 844a6e4
- `~/.claude/skills/` 有 12 个技能,`~/.claude/projects/*/memory/` 有 MEMORY.md
- 模型权重在位:`~/cybernationlism_in_Wuchang/results/finetune/roberta_fold0/model.safetensors`
- 大语料在位:`~/Course_Coding/Cogs206_MachineLearnin/01_data/01_RawCorpus/*.csv`
- conda 环境 nlp 可用,`import torch, transformers, pandas` 不报错
