# 本地预计算排序结果

本目录仅保存在本机，不属于团队 GitHub PR。

四个 JSONL 文件分别保存公开 200 条和合成 3,021 条的 Top50/Top10。有且仅有 `sample_id` 与按排名排列的 `parent_asins` 两个字段；不含 target、ground truth、scenario、profile、质量层级或 evaluator 隐藏字段。

文件校验：

| 文件 | 行数 | 每行候选数 | SHA256 |
|---|---:|---:|---|
| `public200_top50.jsonl` | 200 | 50 | `32B1A1BE0EDB82025F927C155CF5DA2C222E199975CCD0FEF825C2E00909392F` |
| `public200_top10.jsonl` | 200 | 10 | `53D7A89769B6CB9BDDDCA3A50B361FBADEF5A522EEFCD79E30FD7CB0F2A0201A` |
| `synthetic3021_top50.jsonl` | 3,021 | 50 | `D060654655BA7412E916E494BFA933C4011743593D3E3C2F4A0FDF300FE4C30D` |
| `synthetic3021_top10.jsonl` | 3,021 | 10 | `C445ADA4135FEFA8791F31F96FFD9304EA512AB8963FC867EEFDB46C3DB3E2F7` |

已验证所有 `parent_asin` 均存在于冻结 50k catalog，且每条 Top10 都是同一 `sample_id` Top50 的子集。
