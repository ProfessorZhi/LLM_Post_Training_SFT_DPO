# 数据构建流程

## 目标不是“堆数据”，而是构造可训练的 FIM 任务

本项目的数据构建目标不是收集尽可能多的源码片段，而是把源码转成适合 `Fill-in-the-Middle` 学习的补全样本。

每条样本都应满足：

```text
complete_code = prefix + middle + suffix
target = middle
task = prefix + suffix -> middle
```

也就是说，模型学的是“在给定前后文的情况下补全中间缺失片段”，而不是普通左到右续写。

## SFT 主线构建思路

SFT 主线的核心流程可以概括为：

1. 从允许使用的源码来源中抽取候选文件
2. 按 FIM 规则切出 `prefix / middle / suffix`
3. 做去重、路径过滤和质量过滤
4. 形成本地 source parquet
5. 在小实验和固定 eval 中对比分支质量
6. 选出最终 adopted 数据集

### 为什么最后选 `V2.5 fn-only filtered`

早期 clean-path 扩量虽然能显著做大样本量，但仍残留两类会直接伤害 FIM 训练质量的问题：

- `repetition_risk`
- `boundary_continuation`

同时，过长 `middle` 也会增加生成不稳定和结构越界风险。

因此最终在 `V2.4 XXL clean-path` 基础上继续收紧，形成 `V2.5 fn-only filtered`：

- 只保留 `function_body` / `method_body`
- 丢弃 `middle_len > 899`
- 丢弃 `repetition_risk`
- 丢弃 `boundary_continuation`

这样做的本质，是用一部分数据规模换更高的任务纯度和更稳定的补全边界。

## DPO 主线构建思路

DPO 不再直接学习“正确答案长什么样”，而是学习“在同一个 FIM 任务下，什么答案更好、什么答案更差”。

因此每条偏好对都必须仍然保持同一个任务：

```text
same prefix + suffix
chosen   = better middle
rejected = worse middle
```

最终主线采用：

- `chosen`：真实 gold middle
- `rejected`：真实模型生成错误为底，再结合 GPT mimic 扩量

这样做的原因是：

- 真实模型错误最贴近后续推理时会出现的问题
- 只靠纯手写或纯合成坏样本，容易偏离真实错误分布
- 只靠少量真实 model-generated rejected，规模又不够

所以最终选择“少量真实模型错误做锚点 + 大量受控 mimic 扩量”的折中路线。

## 关键质量闸门

无论是 SFT 还是 DPO，进入最终训练池前都至少要过下面这些检查。

### SFT 检查

- 语言与文件类型分布是否失衡
- `middle` 长度是否过长
- 是否混入 test/example/generated/vendor/build/dist/config/setup 等高风险路径
- 是否存在重复样本、模板样本或自动生成文件
- 抽样后是否真的像“值得学习的代码补全任务”

### DPO 检查

- `chosen/rejected` 是否仍属于同一个 `prefix + suffix -> middle` 任务
- `rejected` 是否像真实补全错误，而不是随机胡写
- `chosen/rejected` 是否过于接近或完全离题
- 是否存在重复 pair、三元组重复、`chosen == rejected`
- `rejected` 中是否泄露 Markdown 包装或 FIM 特殊 token

## 为什么公开仓库不直接放入原始数据文件

原因不是“数据不重要”，恰恰相反，是因为数据很重要，所以要用更适合审阅的方式呈现。

公开仓库更适合放：

- 数据集说明
- 最终 adopted manifest
- 数据构建方法
- 关键统计和过滤逻辑

而不适合直接堆：

- 大 parquet 文件
- 大 json 训练文件
- 中间清洗缓存
- 批量审计产物全集

真正的数据原件应由复现者按数据授权自行准备；本仓库只保留构建方法、过滤逻辑和代表性统计，避免公开不必要的大文件或受限数据。
