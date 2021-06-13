[![PyPI - License](https://img.shields.io/hexpm/l/plug)](https://github.com/PrithivirajDamodaran/Styleformer/blob/main/LICENSE)


<p align="center">
    <img src="images/Styleformer.png" width="65%" height="60%"/>
</p>

# Styleformer
A Neural-Style Transfer framework to transfer natural language text smoothly between fine-grained language styles like formal/casual, active/passive and many more. For instance, understand [What makes text formal or casual/informal](https://www.niu.edu/writingtutorial/style/formal-and-informal-style.shtml).

## Table of contents
- [Usecases for Styleformer](#usecases-for-styleformer)
- [Installation](#installation)
- [Quick Start](#quick-start)
  * [Casual to Formal (Available now !)](#casual-to-formal--available-now---)
- [Models](#models)
- [Dataset](#dataset)
- [Benchmark](#benchmark)
- [References](#references)
- [Citation](#citation)

## Usecases for Styleformer

**Area 1: Data Augmentation**
- Augment training datasets with various fine-grained language styles.

**Area 2: Post-processing**
- Apply style transfers to machine generated text. 
- e.g.
    - Refine a Summarised text to active voice + formal tone.
    - Refine a Translated text to more casual tone to reach younger audience.

**Area 3: Assisted writing**
- Integrate this to any human writing interfaces like email clients, messaging tools or social media post authoring tools. Your creativity is your limit to te uses. 
- e.g.
    - Polish an email with business tone for professional uses.

## Installation
```python
pip install git+https://github.com/PrithivirajDamodaran/Styleformer.git
```
## Quick Start

### Casual to Formal (Available now !)
```python
from styleformer import Styleformer

# style = [0=Casual to Formal, 1=Formal to Casual, 2=Active to Passive, 3=Passive to Active]
# inference_on = [0= On CPU, 1= Regular model On GPU, 2=Quantized model On CPU]
sf = Styleformer(style = 3, inference_on=0) 

source_sentences = [
]   

for source_sentence in source_sentences:
    target_sentence = sf.transfer(source_sentence)
    print("[Informal] ", source_sentence)
    print("[Formal] ",target_sentence[0])
    print("-" *100)
```

## Models

|      Model          |Type                          |Status                         
|----------------|-------------------------------|-----------------------------|
|prithivida/informal_to_formal_styletransfer |Seq2Seq |Beta
|prithivida/formal_to_informal_styletransfer|Seq2Seq    |WIP|
|prithivida/active_to_passive_styletransfer|Seq2Seq    |WIP|
|prithivida/passive_to_active_styletransfer|Seq2Seq    |WIP|
|prithivida/positive_to_negative_styletransfer|Seq2Seq    |WIP|
|prithivida/negative_to_positive_styletransfer|Seq2Seq    |WIP|


## Dataset
- TBD
- Fined tuned on T5 on a Tesla T4 GPU and it took ~2 hours to train each of the above models with batch_size = 16 and epochs = 5.(Will share training args shortly)

## Benchmark
- TBD

## References
- [Generative Text Style Transfer for Improved Language Sophistication](http://cs230.stanford.edu/projects_winter_2020/reports/32069807.pdf)
- [Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/pdf/1804.06437.pdf)

## Citation
- TBD




