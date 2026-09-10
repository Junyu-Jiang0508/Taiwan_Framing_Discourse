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

## 重建要点

- WSL 用户名沿用 `jain_farstrider`,否则归档里的绝对路径要额外处理
- 还原脚本在盘根 `F:\scripts\`,不需要先解包
- **conda 环境必须用 `conda-nlp-full.yml`**(28 conda + 223 pip)。
  `conda-nlp-history.yml` 只有 python 和 pip 两项,是空的,不能用。
- 目标版本:Python 3.11.15、Node v24.14.1、R 4.3.3、Ubuntu 24.04
- R 有 244 个包,apt 25 个手动安装包(关键是 r-base-core 和中文 LaTeX 那几个)
- SSH 私钥还原后必须 `chmod 600`,否则 git 推不动

## 三个已知陷阱

1. **活的 Zotero 库在 `D:\Docs\Papers\Zotero`**,不是用户目录下那个。
   `TRANSFER\stale\Zotero-CDrive-DEAD` 是旧机 3 月的失效副本,不要还原。
2. **`D:\Projects` 是 3 至 4 月的过期快照**,WSL 侧才是权威,不要反向覆盖。
3. **Danmaku 的 .md 文档按仓库策略不入 Git**,只存在于归档包里。

## 遗留待办

- 归档区媒体主动放弃于 90%,源文件仍在旧机 D 盘 E 盘
- My-Web-V2 备份因路径写错被跳过,脚本已修
- `D:\下载\Compressed` 首轮有文件复制失败(exit 11),疑似文件名过长,未核对

## 验证清单

还原完成后逐项跑通,全过再考虑清理旧机:

- `ssh -T git@github.com` 能认证
- 六个仓库 dirty=0,Taiwan 项目最新提交为 844a6e4
- `~/.claude/skills/` 有 12 个技能,`~/.claude/projects/*/memory/` 有 MEMORY.md
- 模型权重在位:`~/cybernationlism_in_Wuchang/results/finetune/roberta_fold0/model.safetensors`
- 大语料在位:`~/Course_Coding/Cogs206_MachineLearnin/01_data/01_RawCorpus/*.csv`
- conda 环境 nlp 可用,`import torch, transformers, pandas` 不报错
