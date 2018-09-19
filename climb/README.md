## 寻找存在小范围波动的数据整体的波峰和波谷

统计波峰和波谷的数据

for example:

```
from climb import Climb, VarClimb

climb1 = Climb(data,20) # 传入数据和步长
climb2 = VarClimb(data, 30) # 传入数据和粗化步长
x, y = climb1.get_peaks_troughs() # 获取波峰和波谷列表
x, y = climb2.get_peaks_troughs() # 获取波峰和波谷列表
```
