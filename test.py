from concurrent.futures import ThreadPoolExecutor

from utils import OnlyPutBlockingQueue

if __name__ == '__main__':
    count = 5
    import time


    class TmpGetter(object):
        def __init__(self):
            self.tmp_dict = OnlyPutBlockingQueue()
            # self.tmp_dict = {}
            # self.pool = pool

        def test(self, i: int):
            time.sleep(count - i)
            self.tmp_dict.put(i, "qwe_" + str(i))
            # self.tmp_dict[i] = "qwe_" + str(i)

        def __call__(self, src):
            self.test(src)
            print(src)

    init = [3, 4]
    t = TmpGetter()

    with ThreadPoolExecutor(20) as executor:
        executor.map(t, init)

    print(t.tmp_dict.queue)

    # async_map(t, init)
    print("result")
    print([t.tmp_dict.queue[i] for i in init])