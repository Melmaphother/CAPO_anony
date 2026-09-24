# Benchmark Data

This directory contains prepared splits and lightweight runtime artifacts used by the training and evaluation recipes. Large benchmark assets are intentionally not bundled with the repository; obtain them from the official project sources listed below and place them under the paths expected by the launch scripts.

## Layout

| Path | Benchmark | Purpose |
|---|---|---|
| `corpus/hotpotqa/` | HotpotQA, 2WikiMultiHopQA, MuSiQue | Multi-hop QA training and evaluation splits |
| `pasa/` | PaSa-style paper search | Paper-search training and evaluation splits |
| `scienceworld/` | ScienceWorld | Text-environment training and evaluation splits |
| `webshop/` | WebShop | Lightweight shopping-environment index and splits |

The ALFWorld recipes expect prepared files under `data/alfworld/`. The full WebShop and HotpotQA retrieval artifacts can be placed under `data/webshop_full/` and `data/corpus/hotpotqa_corpus/`, respectively, or supplied through the environment variables documented in the launch scripts.

## Optional retrieval index rebuild

To rebuild a 2WikiMultiHopQA index with `BAAI/bge-large-en-v1.5`:

```bash
python recipe/hotpotqa/process_hotpotqa.py \
  --data_dir data/corpus/2wikimultihopqa_corpus \
  --corpus_path data/corpus/2wikimultihopqa_corpus/hpqa_corpus.jsonl \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --devices cuda:0 \
  --batch_size 256
```

## Data licenses

The datasets remain subject to their original terms:

- [ALFWorld](https://github.com/alfworld/alfworld)
- [WebShop](https://github.com/princeton-nlp/WebShop)
- [HotpotQA](https://hotpotqa.github.io/)
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
- [MuSiQue](https://github.com/StonyBrookNLP/musique)
- [PaSa](https://github.com/bytedance/pasa)
- [ScienceWorld](https://github.com/allenai/ScienceWorld)
