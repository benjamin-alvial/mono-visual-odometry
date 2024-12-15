import os

if __name__ == "__main__":

    for seq in ['00','01','02','03','04','05','06','07','08','09','10']:
        os.system("python run.py "+seq+" False")
    for seq in ['00','01','02','03','04','05','06','07','08','09','10']:
        os.system("python run.py "+seq+" True")