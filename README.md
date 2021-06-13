[![PyPI - License](https://img.shields.io/hexpm/l/plug)](https://github.com/PrithivirajDamodaran/Styleformer/blob/main/LICENSE)


<p align="center">
    <img src="images/Styleformer.png" width="65%" height="60%"/>
</p>

# Styleformer
A Neural-Style Transfer framework to transfer smoothly between formal and casual natural language text renderings. Created by Prithiviraj Damodaran. Open to pull requests and other forms of collaboration.

## Formal and casual language in English
- [What makes text formal or casual/informal](https://www.niu.edu/writingtutorial/style/formal-and-informal-style.shtml)

## Usecases for Styleformer

**Area 1: Data Augmentation**
- Augment training datasets with various fine-grained language styles.

**Area 2: Post-processing**
- Apply style transfers to machine generated text. 
- e.g.
    - Refine a Summarised text to active voice + formal language.
    - Refine a Translated text to more casual to reach younger audience.

**Area 3: Assisted writing**
- Integrate this to any human writing interfaces like email clients, messaging tools or social media post authoring tools. Your creativity is your limit to te uses. 
- e.g.
    - Polish an email with business language for official purposes.

## Installation
```python
pip install git+https://github.com/PrithivirajDamodaran/Styleformer.git
```
## Quick Start

### Casual to Formal (Available now !)
```python
from styleformer import Styleformer

sf = Styleformer(style = 3 , use_gpu=False) # 0=Casual to Formal, 1=Formal to Casual, 2=Active to Passive, 3=Passive to Active, 

source_sentences = [
]   

for source_sentence in source_sentences:
    target_sentence = sf.transfer(source_sentence)
    print("[Informal] ", source_sentence)
    print("[Formal] ",target_sentence[0])
    print("-" *100)
```

