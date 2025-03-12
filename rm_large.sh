# 从 Git 历史中删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch ckpt/sft_qwen_gsm8k/model.safetensors" \
  --prune-empty --tag-name-filter cat -- --all

# 清理和回收空间
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now