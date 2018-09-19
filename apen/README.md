## 计算单通道数据的近似熵和互近似熵

近似熵和互近似熵主要用来分析数据的复杂度

for example:

```
from apen import EegApEn

eegapen = EegApEn(number=1500) # number代表取样的点数
apen1 = eegapen.get_eeg_new_apen(U) # 计算近似熵
apen2 = eegapen.get_eeg_new_huapen(U, G) # 计算互近似熵
```
