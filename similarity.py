import time
from utils import load_data


def main():

    train_csr, dftest = load_data()

    t1 = time.time()
    results = train_csr.dot(train_csr.transpose())
    t2 = time.time()
    print("Dot product of results -- time: %.2f sec" % (t2 - t1))


if __name__ == "__main__":
    main()