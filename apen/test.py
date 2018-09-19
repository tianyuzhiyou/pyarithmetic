from utils.apen import PoolEegApEn, EegApEn
import numpy as np

def test_apen():
    """
    ²âÊÔ¼ÆËã½üËÆìØ
    :return:
    """
    filename = './eeg_all.txt'
    with open(filename) as f:
        eeg_l = f.readlines()
        for eeg in eeg_l:
            eeg = eval(eeg)
            U = np.array(eeg)
            ap = PoolEegApEn(number=1500)
            jss = ap.get_pool_eeg_apen(U)
            print(jss)

def test_huapen():
    """
    ²âÊÔ¼ÆËã»¥½üËÆìØ
    :return:
    """
    filename = './eeg_all.txt'
    with open(filename) as f:
        eeg_l = f.readlines()
        for eeg in eeg_l:
            eeg = eval(eeg)
            U = np.array(eeg)
            ap = EegApEn(number=1500)
            jss = ap.get_eeg_huapen_b(U,U)
            print(jss)

if __name__ == "__main__":
    test_apen()
    test_huapen()