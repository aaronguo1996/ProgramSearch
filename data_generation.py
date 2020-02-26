#robut_data.py
from multiprocessing import Queue, Process

#from pathos.multiprocessing import Queue, Process 
#parallel data gen
import dill

def get_supervised_batchsize(fn, batchsize=200):
    #takes a generation function and outputs lists of optimal size
    remainder = [], []
    while True:
        preS, preA = remainder
        S, A = fn()
        S, A = preS+S, preA+A
        ln = len(S)

        if ln > batchsize:
            yield S[:batchsize], A[:batchsize]
            remainder = S[batchsize:], A[batchsize:]
            continue
        elif ln < batchsize:
            remainder = S, A
            continue
        elif ln == batchsize:
            yield S, A
            remainder = [], []
            continue
        else: assert 0, "uh oh, not a good place"

class GenData:
    def __init__(self, fn, n_processes=20, max_size=100, batchsize=200):
        ##what needs to happen:
        def consumer(Q):
            iterator = get_supervised_batchsize(fn, batchsize=batchsize)
            while True:
                try:
                    # get a new message
                    size = Q.qsize()
                    #print(size)
                    if size < max_size:
                        # process the data
                        ret = next(iterator)
                        Q.put( ret )
                except ValueError as e:
                    print("I think you closed the thing while it was running, but that's okay")
                    break
                except Exception as e:
                    print("error!", e)
                    break

        self.Q = Queue()
        print("started queue ...")

        # instantiate workers
        self.workers = [Process(target=consumer, args=(self.Q,))
               for i in range(n_processes)]

        for w in self.workers:
            w.start()
        print("started parallel workers, ready to work!")

    def batchIterator(self):
        while True:
            yield self.Q.get()
        #yield from get_supervised_batchsize(self.Q.get, batchsize=batchsize) #is this a slow way of doing this??

    def kill(self):
        #KILL stuff
        # tell all workers, no more data (one msg for each)
        # join on the workers
        for w in self.workers:
            try:
                w.close() #this will cause a valueError apparently??
            except ValueError:
                print("killed a worker")
                continue

