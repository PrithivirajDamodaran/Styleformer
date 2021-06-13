[![PyPI - License](https://img.shields.io/hexpm/l/plug)](https://github.com/PrithivirajDamodaran/Styleformer/blob/main/LICENSE)
[![Visits Badge](https://badges.pufler.dev/visits/PrithivirajDamodaran/Styleformer)](https://badges.pufler.dev)

<p align="center">
    <img src="images/Styleformer.png" width="35%" height="35%"/>
</p>

# Styleformer
A Neural-Style Transfer framework to transfer smoothly between formal and casual natural language text renderings. Created by Prithiviraj Damodaran. Open to pull requests and other forms of collaboration.

## Formal and casual language in English
- [What makes text formal or casual/informal](https://www.niu.edu/writingtutorial/style/formal-and-informal-style.shtml)

## Installation
```python
pip install git+https://github.com/PrithivirajDamodaran/Styleformer.git
```
## Quick Start

### Transfer - [Available now]
```python
from styleformer import Styleformer
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)


sf = Styleformer(style = 3 , use_gpu=False) # 0=Active to Passive, 1=Passive to Active, 2=Formal to Informal, 3=Informal to Formal 

source_sentences = [
    "Matt like fish",
    "the collection of letters was original used by the ancient Romans",
    "We enjoys horror movies",
    "Anna and Mike is going skiing",
    "I walk to the store and I bought milk",
    "We all eat the fish and then made dessert",
    "I will eat fish for dinner and drank milk",
    "what be the reason for everyone leave the company",
]   

for source_sentence in source_sentences:
    target_sentence = sf.transfer(source_sentence)
    print("[Informal] ", source_sentence)
    print("[Formal] ",target_sentence[0])
    print("-" *100)
```

