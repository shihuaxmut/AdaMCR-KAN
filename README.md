# AdaMCR-KAN
![Figure1_framework](Figure1_framework.png)

## Abstract
### English
TCR–pMHC-I recognition is governed by intricate residue-wise cross-molecular interactions and sequence dependencies, and the extreme polymorphism of HLA-I molecules brings great difficulty to accurate prediction. We propose AdaMCR-KAN, a KAN-based model for TCR–pMHC-I recognition prediction. It uses a hierarchical residue interaction encoder (HRIEncoder) to extract multi-level sequence features, bidirectional cross-attention with adaptive head gating, and a novel cross-interaction gating (CIG) module to dynamically filter redundant interactions and emphasize key binding patterns. Benchmark results show AdaMCR-KAN achieves superior performance and strong generalization across diverse peptides, HLA alleles and test scenarios. Further motif analysis and saturation mutagenesis validation confirm the model learns biologically meaningful residue binding preferences.
