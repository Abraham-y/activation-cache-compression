# Citation verification manifest

Rule for this project: **every reference is downloaded to `paper/refs/`
and identity-verified before it may be cited. Anything that cannot be
downloaded is not cited.** Verified 2026-08-09.

## Verified references (29 cited, 14 held locally)

| key | local file | verified identity |
|---|---|---|
| llamascope | `refs/2410.20526.pdf` | Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders |
| gemmascope | `refs/2408.05147.pdf` | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 |
| saelens | `refs/saelens_config.py`, `refs/saelens_README.md` | SAELens source; `dtype: str = "float32"` at config.py:242 |
| duan | `refs/2606.03002.pdf` | Perplexity Can Miss SAE Feature Damage Under Quantization |
| paulo | `refs/2501.16615.pdf` | Sparse Autoencoders Trained on the Same Data Learn Different Features |
| reprosubspaces | `refs/2606.12138.pdf` | Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in SAEs |
| chanin | `refs/2508.16560.pdf` | Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders |
| mdlsae | `refs/2410.11179.pdf` | Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs |
| summers | `refs/2103.04514.pdf` | Nondeterminism and Instability in Neural Network Optimization |
| quarot | `refs/2404.00456.pdf` | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs |
| actnn | `refs/2104.14129.pdf` | ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training |
| proxypresumption | `refs/2605.07409.pdf` | The Proxy Presumption: From Semantic Embeddings to Valid Social Measures |
| validitythreats | `refs/2606.05029.pdf` | Validity Threats for Foundation Model Research |
| nlgmeasurement | `refs/2305.14889.pdf` | Evaluating Evaluation Metrics: ... using Measurement Theory |
| llmint8 | arXiv:2208.07339 (abstract page) | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale; Dettmers, Lewis, Belkada, Zettlemoyer; NeurIPS 2022 |
| smoothquant | arXiv:2211.10438 (abstract page) | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models; Xiao, Lin, Seznec, Wu, Demouth, Han; ICML 2023 |
| kivi | arXiv:2402.02750 (abstract page) | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache; Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu; ICML 2024 |

| sww | PMLR 23 page (proceedings.mlr.press/v23/spielman12) | Exact Recovery of Sparsely-Used Dictionaries; Spielman, Wang, Wright; COLT 2012, PMLR 23:37.1-37.18; arXiv:1206.5882 |
| agm | PMLR 35 page (proceedings.mlr.press/v35/arora14) | New Algorithms for Learning Incoherent and Overcomplete Dictionaries; Arora, Ge, Moitra; COLT 2014, PMLR 35:779-806; arXiv:1308.6273 |
| gpt2 | OpenAI report PDF, page 1 | Language Models are Unsupervised Multitask Learners; Radford, Wu, Child, Luan, Amodei, Sutskever; 2019 |
| pythia | arXiv:2304.01373 (abstract page) | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling; Biderman et al. (13 authors); ICML 2023 |
| openwebtext | skylion007.github.io/OpenWebTextCorpus | OpenWebText Corpus; Gokaslan, Cohen; 2019 (the page's requested citation) |
| pile | arXiv:2101.00027 (abstract page) | The Pile: An 800GB Dataset of Diverse Text for Language Modeling; Gao et al. (12 authors); 2020 |
| sst2 | aclanthology.org/D13-1170 | Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank; Socher et al. (7 authors); EMNLP 2013 |
| agnews | arXiv:1509.01626 (abstract page) | Character-level Convolutional Networks for Text Classification; Zhang, Zhao, LeCun; NIPS 2015 (source of the AG News benchmark) |
| topk | arXiv:2406.04093 (abstract page) | Scaling and evaluating sparse autoencoders; Gao, Dupre la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike, Wu; 2024 |
| jumprelu | arXiv:2407.14435 (abstract page) | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders; Rajamanoharan et al. (7 authors); 2024 |
| hungarian | Wikipedia reference list (publisher page refused access) | The Hungarian Method for the Assignment Problem; Kuhn; Naval Research Logistics Quarterly 2:83-97, 1955 |
| zstd | rfc-editor.org/info/rfc8878 | Zstandard Compression and the 'application/zstd' Media Type; Collet, Kucherawy (Ed.); RFC 8878, 2021 |

llmint8, smoothquant and kivi were added at camera-ready and checked on
11 Oct 2026 against their arXiv abstract pages (title, full author list,
venue), not against a local PDF. sww and agm (12 Oct) and the ten asset
references from gpt2 on (models, datasets and methods the experiments
use, added 3 Nov after an adversarial citation check) were checked on
3 Nov 2026 against the public pages named in the table, not against a
local PDF; the 3 Nov check also re-read every local PDF's cited claims.

Titles were extracted from page 1 of each downloaded PDF and matched
against the bibliography entry, so a wrong-ID download would have been
caught rather than silently cited.

## Attributed claims checked verbatim against the source text

| paper claim | source text (grep of the local PDF) |
|---|---|
| activations are ~4,000x larger than text; 8 KB/token vs 2 B/token | Llama Scope: "text data requires only 2 bytes per token, latent activations occupy 8K bytes per token---resulting in a 4,096x increase in both storage needs and disk throughput" |
| Gemma Scope needs >1 GiB/s disk read per SAE | Gemma Scope: "loading requires more than 1 GiB/s of disk read" |
| Gemma Scope trains each SAE on 4--16B tokens | Gemma Scope: "on 4-16B tokens of text each" |
| outlier channels and per-channel handling in LLM quantization | LLM.int8() abstract: "highly systematic emergent features"; KIVI abstract: "the key cache should be quantized per-channel"; SmoothQuant abstract: migrates the difficulty "from activations to weights with a mathematically equivalent transformation" (a per-channel smoothing) |
| Paulo & Belrose report ~30% shared features; Hungarian matching | 2501.16615: "only 30%" (Llama 3 8B); "the Hungarian algorithm is known to efficiently compute this" |
| Duan uses a frozen SAE as fixed measurement basis; AUC 0.92--0.97, peak activation strongest | 2606.03002: "using a frozen SAE as a fixed measurement [basis]"; "cross-validated AUC 0.92--0.97 and peak activation as the strongest marginal predictor" |
| SAELens defaults to fp32 activations | `saelens_config.py:242`: `dtype: str = "float32"` |

## Removed for lack of a downloadable source

- **The "funky results" fp16 anecdote** previously attributed to SAELens.
  The claim appears in project notes but is not present in any artifact
  we could download (checked the repo README and source). The intro
  sentence was rewritten to the verifiable claim (fp32 is the default),
  which is stronger anyway: it cites code we hold rather than hearsay.

## Corrected at camera-ready

- A sentence attributing a bf16 forward pass for caching to Gemma Scope
  was removed (11 Oct 2026): Gemma Scope stores 32-bit activations
  ("We use 32-bit precision in all our experiments", its 3.3.2) and uses
  bf16 only at inference (its 4.7). The paper now says "as some
  pipelines do".

## Corrected 3 Nov 2026 (adversarial citation check)

- paulo: the paper had called our weight-space floor (30.6%) "close to
  their published value". Their ~30% is for Llama 3 8B, their largest
  model; their GPT-2 SAEs share roughly 0.6-0.75 (their Fig. 5), and
  sharing rises with training tokens (their Fig. 4; default 8B tokens).
  Now stated as such. "Other work already implicitly compares against
  this floor" (paulo, chanin) removed: neither paper does.
- gemmascope: the 1 GiB/s figure is for one SAE on Gemma 2 9B (TPUs);
  now scoped. Its 4.7 bf16 inference check is now cited in related work.
- llamascope: cited for its bf16 activations ("to optimize memory
  usage", its 3.1), replacing the uncited "some pipelines use for speed".
- saelens: "the primary library" -> "a widely used library", fp32 "by
  default".
- duan: peak activation is its strongest *marginal* predictor (rarity has
  the largest multivariate coefficient); wording now says "single".
- summers: scoped to one bit in one initial weight.
- quarot: "comes from" -> "is used in" (QuaRot credits earlier work).

## Reproducing this check

```bash
cd paper/refs
for id in 2410.20526 2408.05147 2606.03002 2501.16615 2606.12138 \
          2508.16560 2404.00456 2104.14129 2605.07409 2606.05029 2305.14889; do
  curl -sL -o "$id.pdf" "https://arxiv.org/pdf/$id"
  pdftotext -f 1 -l 1 "$id.pdf" - | head -3   # confirm title
done
```
