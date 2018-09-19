"""
通过斜率的平均值来寻找波峰和波谷，对于一段在局部可能存在波动，但是在总体上呈现波动，通过多个点的平均值
"""
import numpy as np

class VarClimb(object):
    """
    通过进行数据滤波后再使用爬坡算法，
    设置一个步长，步长的点都用平均值来替代，从而减少某个点的波动造成的波峰
    """
    def __init__(self, data:list, step: int = 5):
        """

        :param data: 一个一维数组
        :param step: 步长
        :param r: 阀值
        """
        self.data = np.array(data)
        self.step = step

    def get_means_array(self):
        """
        对数据进行粗化
        :return:
        """
        l = self.data.size//self.step
        self.data = self.data[:l * self.step]
        arr = np.reshape(self.data, (l,self.step))
        for i, a in enumerate(arr):
            self.data[i * self.step:(i + 1) * self.step] = np.array([np.sum(a)/self.step for j in range(self.step)])

        return self.data

    def get_peaks_troughs(self):
        """
        获取波峰和波谷列表
        :return:
        """
        data = self.get_means_array()
        return Climb(data,self.step).get_peaks_troughs()


class Climb(object):
    """
    多点获取平均值计算波峰波谷，
    获取n到n+step的点的平均值与n+1到n+step+1的点的平均值进行比较，转折点判断波峰和波谷，可以去掉干扰。
    """

    def __init__(self, data: list, step: int = 5):
        """
        :param data: 一个一维数组
        :param step: 步长
        """
        self.data = data
        self.step = step
        self.peaks = list()  # 波峰列表
        self.troughs = list()  # 波谷列表
        self.S = list()  # 用来标记当前是上坡还是下坡，1上坡，2下坡
        self.array = None

    def _get_array_two(self):
        """
        将data转化为二维矩阵
        :return:
        """

        return np.array([self.data[i:i+self.step] for i in range(len(self.data)-self.step + 1)])

    def _get_list_S(self):
        """
        获取上下坡的列表
        :return:
        """
        if self.S:
            return self.S
        np_data = self._get_array_two()
        self.S.extend([1 if np.sum(np_data[i + 1] - np_data[i]) >= 0 else 2 for i in range(np_data.shape[0] - 1)])
        return self.S

    def get_peaks_troughs(self):
        """
        获取波峰和波谷列表
        :return:
        """
        S = self._get_list_S()
        if not self.array:
            self.array = self._get_array_two()
        for i in range(len(S)-1):
            if S[i + 1] > S[i]:
                self.peaks.append(np.median(self.array[i + 1]))
            elif S[i + 1] < S[i]:
                self.troughs.append(np.median(self.array[i + 1]))
        print(self.data)

        return self.peaks, self.troughs

    def get_number_peaks_troughs(self):
        """
        获取波峰和波谷的数量
        :return:
        """
        self.get_peaks_troughs()
        return len(self.peaks), len(self.troughs)




if __name__ == '__main__':
    pass
