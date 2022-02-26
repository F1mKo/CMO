import multiprocessing
import time
import random
import numpy as np

def server(input_q,next_q,i):
    while True:
        item = input_q.get()
        if i==0:item.st=time.time() ## start recording time
                                                    ## (first phase)
        time.sleep(random.expovariate(glambda[i]))
##stop recording time (last phase)
        if i==M-1 :item.st=time.time()-item.st
        next_q.put(item)
        input_q.task_done()
    print("Server%d stop" % i) ##will be never printed why?

def producer(sequence,output_q):
    for item in sequence:
        time.sleep(random.expovariate(glambda[0]))
        output_q.put(item)

def consumer(input_q):
    "Finalizing procedures"
    ## start recording processing time
    ptime=time.time()
    in_seq=[]
    while True:
        item = input_q.get()
        in_seq+=[item]
        input_q.task_done()
        if item.cid == N-1:
            break
    print_results(in_seq)
    print("END")
    print("Processing time sec. %d" %(time.time()-ptime))
    ## stop recording processing time
    print("CPU used %d" %(multiprocessing.cpu_count()))

def print_results(in_seq):
    "Output rezults"
    f=open("out.txt","w")
    f.write("%d\n" % N)
    for t in range(M):
        f.write("%d%s" % (glambda[t],","))
    f.write("%d\n" % glambda[M])

    for t in range(N-1):
        f.write("%f%s" % (in_seq[t].st,","))
    f.write("%f\n" % (in_seq[N-1].st))
    f.close()

class Client(object):
    "Class client"
    def __init__(self,cid,st):
        self.cid=cid ## customer id
        self.st=st ## sojourn time of the customer

###GLOBALS
N=100 ## total number of customers arrived
M=5 ## number of servers
### glambda - arrival + servicing frequency
### = customers/per time unit
glambda=np.array([30000]+[i for i in np.linspace(25000,5000,M)])
all_clients=[Client(num,0) for num in range(0,N)]

###START
if __name__ == "__main__":
        q = [multiprocessing.JoinableQueue() for i in range(M + 1)]
        for i in range(M):
            serv = multiprocessing.Process(target=server,args=(q[i],q[i+1],i))
            serv.daemon=True
            serv.start()
        cons = multiprocessing.Process(target=consumer,args=(q[M],))
        cons.start()

        ### start 'produsing' customers
        producer(all_clients,q[0])

        for i in q:
            i.join()
