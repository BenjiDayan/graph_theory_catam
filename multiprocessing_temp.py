# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 08:26:37 2020

@author: benja
"""
from pathos.multiprocessing import ProcessingPool as Pool
import random
import time
import tqdm

def asdf(a):
    return(a+1)
    
def foo(a):
    import time
    import random
    from multiprocessing_temp import asdf
    time.sleep(1)
    return(asdf(a + random.random()))
    # return(a + random.random())
    
def goo():
    pool = Pool(4)
#    def f(x):
#        return foo(100 + x)
    stuff = list(tqdm.tqdm(pool.imap(foo, range(20)), total=20))
    print(stuff)
    print('aaa')
    pool.close()
    pool.join()
    print('bbb')

if __name__ == '__main__':
#    pool = Pool(4)
#    stuff = list(tqdm.tqdm(pool.imap(foo, range(20)), total=20))
#    print(stuff)
#    print('aaa')
#    pool.close()
#    pool.join()
#    print('bbb')
     goo()