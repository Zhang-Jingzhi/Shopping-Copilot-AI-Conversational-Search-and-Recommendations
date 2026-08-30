# 3,021 条会话扩展数据包

打开 `report.html` 阅读中文技术报告；先读 `manifest.json` 了解文件校验与上游引用。

## 包含的数据

- `data/synthetic_contract_matched_all_3021.jsonl`：官方公开 schema 兼容的 3,021 条本地扩展会话。
- `data/synthetic_contract_matched_all_3021_tiers.jsonl`：以 `sample_id` 关联的质量层级。
- `data/product_filter_inference_3021.csv`：按 `parent_asin` 的筛选共识频率及原始代理信号。
- `data/schema_alignment.json`：逐字段公开 200 对齐说明。
- `data/construction_lineage.json`：来源、步骤、边界和外部输入校验值。

## 重要边界

这是本地正未标注代理集，不是 organizer private 800 数据，也不能将低可能层当作负例。`preference_tags` 被刻意留空，因为真实用户来源与标签算法未公开。`average_prior_rating` 仅维持公开接口的类型、值域和边际分布；它使用目标 test 单次评分代理，不能视为用户历史平均评分。

它适合用于压力测试、bad-case 分析和代理训练实验。任何在这 3,021 条上训练并在同一批数据上测得的结果都必须标为 full-fit 结果，不能当作泛化成绩或私有测试预测。

## 推送前核验

- 主会话文件：3,021 行、3,021 个唯一 `sample_id`、3,021 个唯一代理目标；
- 与公开 200 条目标商品交集：0；
- `preference_tags`：3,021/3,021 为空；
- scenario：1,209 buying、1,208 browsing、453 intent override、151 boundary；
- `manifest.json` 所列文件 SHA256：全部匹配。

若修改任何数据文件，请同步更新 `manifest.json`、construction lineage 和技术报告，并在 PR 中说明生成脚本、输入校验值及行数变化。
