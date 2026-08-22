# save 权重与 checkpoint 清理记录

## 目标

清理 `/home/rpartx3080/CodeSpace/ReGenNet/save` 中用户明确允许删除的模型权重与 checkpoint，释放磁盘空间，同时保留实验配置和结果记录。

## 范围确认

- 目标文件：`save/**/*.pt`。
- `.pt` 文件包含模型参数、优化器状态、normalizer、classifier 等可加载权重或 checkpoint 内容。
- 已确认共 2,579 个 `.pt` 文件，约 91.48GiB。
- 保留 `args.json`、指标 JSON/JSONL/YAML、日志、PID、退出标记和 `.npy` 诊断文件。
- 删除前检查到 PID 文件，但对应训练进程均未运行，不会中断训练。

## 执行与复核

使用按路径限定的 `find save -type f -name '*.pt' -delete` 删除目标；完成后重新统计 `.pt` 文件数量、`save` 占用和剩余文件类型。结果追加在本文件末尾，不覆盖历史上下文。
