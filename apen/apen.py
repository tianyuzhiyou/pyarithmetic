"""
计算近似熵的工具模块
"""
import numpy as np
from multiprocessing import Pool


class BaseApEn(object):
    """
    近似熵基础类
    """

    def __init__(self, m, r):
        """
        初始化
        :param U:一个矩阵列表，for example:
            U = np.array([85, 80, 89] * 17)
        :param m: 子集的大小，int
        :param r: 阀值基数，0.1---0.2
        """
        self.m = m
        self.r = r

    @staticmethod
    def _maxdist(x_i, x_j):
        """计算矢量之间的距离"""
        return np.max([np.abs(np.array(x_i) - np.array(x_j))])

    @staticmethod
    def _biaozhuncha(U):
        """
        计算标准差的函数
        :param U:
        :return:
        """
        if not isinstance(U, np.ndarray):
            U = np.array(U)
        return np.std(U, ddof=1)


class ApEn(BaseApEn):
    """
    Pincus提出的算法，计算近似熵的类
    """

    def _biaozhunhua(self, U):
        """
        将数据标准化，
        获取平均值
        所有值减去平均值除以标准差
        """
        self.me = np.mean(U)
        self.biao = self._biaozhuncha(U)
        print(self.biao)
        res = np.array([(x - self.me) / self.biao for x in U])
        print(res)
        return res

    def _dazhi(self, U):
        """
        获取阀值
        :param U:
        :return:
        """
        if not hasattr(self, "f"):
            self.f = self._biaozhuncha(U) * self.r
        return self.f

    def _phi(self, m, U):
        """
        计算熵值
        :param U:
        :param m:
        :return:
        """
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self._dazhi(U)]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        sha = np.sum(np.log(list(filter(lambda a: a, C)))) / (len(U) - m + 1.0)
        return sha

    def _phi_b(self, m, U):
        """
        标准化数据计算熵值
        :param m:
        :param U:
        :return:
        """
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self.r]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x: x, C)))) / (len(U) - m + 1.0)

    def jinshishang(self, U):
        """
        计算近似熵
        :return:
        """
        return np.abs(self._phi(self.m + 1, U) - self._phi(self.m, U))

    def jinshishangbiao(self, U):
        """
        将原始数据标准化后的近似熵
        :param U:
        :return:
        """
        eeg = self._biaozhunhua(U)
        return np.abs(self._phi_b(self.m + 1, eeg) - self._phi_b(self.m, eeg))


class FileApEn(ApEn):
    """
    打开文件流分析近似熵和基于二进制流分析
    """

    def anal_file(self, filename, mode='r', encod='utf-8', shi=0):
        """
        打开eeg的文件读取eeg数据
        :param filename:
        :param mode:
        :param shi:
        :param encod:
        :return:
        """
        with open(filename, mode=mode, encoding=encod) as f:
            eeg = f.read()
            # 过滤转化矩阵
            U = np.array([float(i) for i in filter(lambda x: x, eeg.split('\n'))])
            if shi == 0:
                return self.jinshishang(U)
            else:
                return self.jinshishangbiao(U)

    def anal_string(self, data: str, shi=0):
        """
        传入字符串形式的eeg数据
        :param shi:
        :param data:
        :return:
        """
        U = np.array([float(i) for i in filter(lambda x: x, data.split('\n'))])
        if shi == 0:
            return self.jinshishang(U)
        else:
            return self.jinshishangbiao(U)


class HuApEn(BaseApEn):

    def _xiefangcha(self, U, G):
        """
        计算协方差的函数
        :param U: 序列1，矩阵
        :param G: 序列2，矩阵
        :return: 协方差，float
        """
        if not isinstance(U, np.ndarray):
            U = np.array(U)
        if not isinstance(G, np.ndarray):
            G = np.array(G)
        if len(U) != len(G):
            raise AttributeError('参数错误！')
        return np.cov(U, G, ddof=1)[0, 1]

    def _biaozhunhua(self, U, G):
        """
        对数据进行标准化
        """
        self.me_u = np.mean(U)
        self.me_g = np.mean(G)
        self.biao_u = self._biaozhuncha(U)
        self.biao_g = self._biaozhuncha(G)
        # self.biao_u = self._xiefangcha(U, G)
        # self.biao_g = self._xiefangcha(U, G)
        return np.array([(x - self.me_u) / self.biao_u for x in U]), np.array(
            [(x - self.me_g) / self.biao_g for x in U])

    def _dazhi(self, U, G):
        """
        获取阀值
        :param r:
        :return:
        """
        if not hasattr(self, "f"):
            self.f = self._xiefangcha(U, G) * self.r
        return self.f

    def _phi(self, m, U, G):
        """
        计算熵值
        :param m:
        :return:
        """
        # 获取X矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取y矢量列表
        y = [G[g:g + m] for g in range(len(G) - m + 1)]
        # 获取所有的条件概率列表
        C = [len([1 for y_k in y if self._maxdist(x_i, y_k) <= self._dazhi(U, G)]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x_1: x_1, C)))) / (len(U) - m + 1.0)

    def _phi_b(self, m, U, G):
        """
        标准化数据计算熵值
        :param m:
        :param U:
        :return:
        """
        # 获取X矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取y矢量列表
        y = [G[g:g + m] for g in range(len(G) - m + 1)]
        # 获取所有的条件概率列表
        C = [len([1 for y_k in y if self._maxdist(x_i, y_k) <= self.r]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x: x, C)))) / (len(U) - m + 1.0)

    def hujinshishang(self, U, G):
        """
        计算互近似熵
        :return:
        """
        return np.abs(self._phi(self.m + 1, U, G) - self._phi(self.m, U, G))

    def hujinshishangbiao(self, U, G):
        """
        将原始数据标准化后的互近似熵
        :param U:
        :param G:
        :return:
        """
        u, g = self._biaozhunhua(U, G)
        return np.abs(self._phi_b(self.m + 1, u, g) - self._phi_b(self.m, u, g))


class NewBaseApen(object):
    """新算法基类"""

    @staticmethod
    def _get_array_zeros(x):
        """
        创建N*N的0矩阵
        :param U:
        :return:
        """
        N = np.size(x, 0)
        return np.zeros((N, N), dtype=int)

    @staticmethod
    def _get_c(z, m):
        """
        计算熵值的算法
        :param z:
        :param m:
        :return:
        """
        N = len(z[0])
        # 概率矩阵C计算
        c = np.zeros((1, N - m + 1))
        if m == 2:
            for j in range(N - m + 1):
                for i in range(N - m + 1):
                    c[0, j] += z[j, i] & z[j + 1, i + 1]
        if m == 3:
            for j in range(N - m + 1):
                for i in range(N - m + 1):
                    c[0, j] += z[j, i] & z[j + 1, i + 1] & z[j + 2, i + 2]
        if m != 2 and m != 3:
            raise AttributeError('m的取值不正确！')
        data = list(filter(lambda x: x, c[0] / (N - m + 1.0)))
        if not all(data):
            return 0
        return np.sum(np.log(data)) / (N - m + 1.0)


class NewHuApEn(HuApEn, NewBaseApen):
    """
    洪波等人提出的快速实用算法计算互近似熵
    """

    def _get_distance_array(self, U, G):
        """
        获取距离矩阵
        :param U:模板数据
        :return:比较数据
        """
        z = self._get_array_zeros(U)
        fa = self._dazhi(U, G)
        for i in range(len(z[0])):
            z[i, :] = (np.abs(G - U[i]) <= fa) + 0
        return z

    def _get_shang(self, m, U, G):
        """
        计算熵值
        :param U:
        :return:
        """
        # 获取距离矩阵
        Z = self._get_distance_array(U, G)
        return self._get_c(Z, m)

    def hongbo_hujinshishang(self, U, G):
        """
        对外的计算互近似熵的接口
        :param U:
        :param G:
        :return:
        """
        return np.abs(self._get_shang(self.m + 1, U, G) - self._get_shang(self.m, U, G))


class NewApEn(ApEn, NewBaseApen):
    """
    洪波等人提出的快速实用算法计算近似熵
    """

    def _get_distance_array(self, U):
        """
        获取距离矩阵
        :param U:
        :return:
        """
        if hasattr(self, 'z'):
            return self.z
        z = self._get_array_zeros(U)
        print(z.shape)
        for i in range((z.shape)[0]):
            z[i, :] = (np.abs(U - U[i]) <= self._dazhi(U)) + 0
        self.z = z
        return self.z

    def _get_shang(self, m, U):
        """
        计算熵值
        :param U:
        :return:
        """
        # 获取距离矩阵
        Z = self._get_distance_array(U)
        return self._get_c(Z, m)

    def hongbo_jinshishang(self, U):
        """
        计算近似熵
        :param U:
        :return:
        """
        return np.abs(self._get_shang(self.m + 1, U) - self._get_shang(self.m, U))


class PoolApEn(NewApEn):
    """
    使用多进程优化近似熵的计算效率
    """

    def __init__(self, m, r, n):
        super().__init__(m, r)
        self.n = n

    def hongbo_jinshishang(self, U):
        """
        重写算法，添加多进程
        :param U:
        :return:
        """
        pool = Pool(self.n)
        ls = pool.starmap(self._get_shang, [(self.m + 1, U), (self.m, U)])
        pool.close()
        pool.join()
        return np.abs(ls[0] - ls[1])


class EegApEn(object):
    """
    eeg数据的近似熵和互近似熵的计算
    """

    def __init__(self, number=1500):
        if number % 2 != 0:
            raise AttributeError('参数必须为偶数！')
        self.number = number  # 设置序列的维度

    def _deal_eeg(self, U):
        """处理eeg数据"""
        if len(U) < self.number:
            raise AttributeError('参数长度不够！')
        start = int(len(U) // 2 - self.number / 2)
        end = int(len(U) // 2 + self.number / 2)
        return U[start:end]

    def get_eeg_apen(self, U):
        """
        获取eeg脑电的近似熵
        :param U:
        :return:
        """
        if not hasattr(self, 'apen'):
            self.apen = ApEn(2, 0.2)
        eeg = self._deal_eeg(U)
        return self.apen.jinshishang(eeg)

    def get_eeg_apen_b(self, U):
        """
        获取eeg脑电的数据标准化后的近似熵
        :param U:
        :return:
        """
        if not hasattr(self, 'apen'):
            self.apen = ApEn(2, 0.2)
        eeg = self._deal_eeg(U)
        return self.apen.jinshishangbiao(eeg)

    def get_eeg_huapen(self, U, G):
        """
        获取eeg脑电的互近似熵
        :param U:
        :param G:
        :return:
        """
        if not hasattr(self, 'huapen'):
            self.huapen = HuApEn(2, 0.2)
        eeg_u = self._deal_eeg(U)
        eeg_g = self._deal_eeg(G)
        return self.huapen.hujinshishang(eeg_u, eeg_g)

    def get_eeg_huapen_b(self, U, G):
        """
        获取eeg脑电的数据标准化后的互近似熵
        :param U:
        :param G:
        :return:
        """
        if not hasattr(self, 'huapen'):
            self.huapen = HuApEn(2, 0.2)
        eeg_u = self._deal_eeg(U)
        eeg_g = self._deal_eeg(G)
        return self.huapen.hujinshishangbiao(eeg_u, eeg_g)

    def get_eeg_new_apen(self, U):
        """
        快速算法计算近似熵,
        :U:必须为一个矩阵数组
        :return:
        """
        if not hasattr(self, 'newapen'):
            self.newapen = NewApEn(2, 0.2)
        eeg = self._deal_eeg(U)
        return self.newapen.hongbo_jinshishang(eeg)

    def get_eeg_new_huapen(self, U, G):
        """
        快速算法计算互近似熵
        :param U:
        :param G:
        :return:
        """
        if not hasattr(self, 'newhuapen'):
            self.newhuapen = NewHuApEn(2, 0.2)
        eeg_u = self._deal_eeg(U)
        eeg_g = self._deal_eeg(G)
        return self.newhuapen.hongbo_hujinshishang(eeg_u, eeg_g)


class PoolEegApEn(EegApEn):
    """多进程计算eeg"""

    def __init__(self, apen=PoolApEn(2, 0.2, 2), number=1500):
        super().__init__(number=number)
        self.apen = apen

    def get_pool_eeg_apen(self, U):
        """
        计算近似熵。
        :param U:
        :return:
        """
        eeg = self._deal_eeg(U)
        return self.apen.hongbo_jinshishang(eeg)


if __name__ == "__main__":
    pass
