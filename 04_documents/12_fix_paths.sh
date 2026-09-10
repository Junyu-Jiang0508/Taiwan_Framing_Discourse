#!/usr/bin/env bash
# 还原后修正硬编码的旧用户名路径。
# 只改「会被执行的」文件:dotfiles、脚本、配置。
# 刻意不碰日志与运行清单 —— 那是历史记录,改写等于伪造过去的运行痕迹。
# 用法: bash 12_fix_paths.sh [--apply] [旧HOME] [新HOME]
set -uo pipefail
APPLY=0
[ "${1:-}" = "--apply" ] && { APPLY=1; shift; }
OLD="${1:-/home/jain_farstrider}"
NEW="${2:-$HOME}"

echo "旧路径: $OLD"
echo "新路径: $NEW"
[ "$OLD" = "$NEW" ] && { echo "两者相同,无需修改。"; exit 0; }
[ "$APPLY" = "1" ] && echo "模式: 实际写入(备份为 .bak-path)" || echo "模式: 干跑"
echo

# 1) 用户级 dotfiles 与配置
FILES=""
for f in "$NEW/.bashrc" "$NEW/.profile" "$NEW/.bash_profile" "$NEW/.gitconfig" \
         "$NEW/.condarc" "$NEW/.claude/settings.json"; do
  [ -f "$f" ] && grep -qF "$OLD" "$f" 2>/dev/null && FILES="$FILES$f"$'\n'
done

# 2) 项目里会被执行的文件。排除日志、清单、输出、数据目录。
SCRIPTS=$(find "$NEW" -maxdepth 6 \
    \( -name .git -o -name node_modules -o -name '.venv*' -o -name venv \
       -o -name miniforge3 -o -name .cursor-server -o -name .vscode-server \
       -o -name .npm -o -name .cache -o -name outputs -o -name 03_outputs \
       -o -path "$NEW/migration/scripts" \) -prune -o \
    \( -name '*.py' -o -name '*.sh' -o -name '*.R' -o -name '*.yaml' -o -name '*.yml' \
       -o -name 'settings.local.json' -o -name 'settings.json' \) -type f -print 2>/dev/null \
  | xargs -r grep -lF "$OLD" 2>/dev/null)
[ -n "$SCRIPTS" ] && FILES="$FILES$SCRIPTS"$'\n'

# 3) 03_outputs 下确实是代码的少数文件(analyze/compute 脚本),单独捞
EXTRA=$(find "$NEW" -path '*/03_outputs/*' -name '*.py' -type f -print 2>/dev/null \
  | xargs -r grep -lF "$OLD" 2>/dev/null)
[ -n "$EXTRA" ] && FILES="$FILES$EXTRA"$'\n'

FILES=$(echo "$FILES" | grep -v '^$' | sort -u)
if [ -z "$FILES" ]; then echo "没有需要修改的可执行文件。"; exit 0; fi

n=0
while IFS= read -r f; do
  [ -f "$f" ] || continue
  c=$(grep -cF "$OLD" "$f" 2>/dev/null || echo 0)
  printf "  %3s 处  %s\n" "$c" "${f#$NEW/}"
  if [ "$APPLY" = "1" ]; then
    cp -p "$f" "$f.bak-path"
    sed -i "s|$OLD|$NEW|g" "$f"
  fi
  n=$((n+1))
done <<< "$FILES"

echo
echo "共 $n 个文件"
echo "(日志与运行清单里的旧路径刻意保留,那是历史记录)"
if [ "$APPLY" = "1" ]; then
  echo
  echo "--- 复查 ---"
  grep -n "miniforge3" "$NEW/.bashrc" 2>/dev/null | head -2
  grep -n "helper" "$NEW/.gitconfig" 2>/dev/null | head -2
  echo
  echo "回滚: find $NEW -name '*.bak-path' | while read b; do mv \"\$b\" \"\${b%.bak-path}\"; done"
else
  echo "确认无误后加 --apply 执行。"
fi
