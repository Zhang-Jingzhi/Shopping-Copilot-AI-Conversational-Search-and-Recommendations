# 3,021 商品排序候选数据

## 数据说明

本包只提供商品级候选 ID、连续弱监督评分和四级分层。

## 数据范围

`ranking_candidates_3021.csv` 共 3,021 行，一行对应一个 `parent_asin` 商品家族。它来自本地 LLO test 商品目标集合中、排除公开 200 个目标后得到的剩余商品；因此是**正未标注候选集合**，不是 organizer private 800 的标签表，也不应把未出现于 private 的商品视为负例。

## 文件

| 文件 | 内容 |
| --- | --- |
| `ranking_candidates_3021.csv` | 3,021 个商品候选及其连续弱监督评分和四级分层。 |

## 三个字段

| 列 | 含义 |
| --- | --- |
| `parent_asin` | 商品家族 ID，用于关联、去重和回查 catalog。 |
| `selection_frequency` | 100 个保留的合理商品筛选模型中该商品被选中的频率；它是相对的弱监督排序分数，不是 private 800 中出现的真实或校准概率。 |
| `quality_tier` | 由筛选共识分桶得到的四层：`high_confidence`（1,065）、`probable`（144）、`uncertain`（168）、`low_likelihood`（1,644）。 |

## 未提供的信息与已知边界

- 不含商品文本、商品属性、用户 ID、时间戳、用户 profile、偏好标签、真实购买历史或 organizer private 800 的真实标签。
- `selection_frequency` 与 `quality_tier` 都是本地弱监督推断的结果，不是官方标注。
- 这份数据可用于研究“哪些商品更符合本地推断的候选池模式”；它不能证明某个商品一定会出现在 private 800 中。
