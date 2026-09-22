from time import time

def manage_time_print(k,tot,prev_time,verbose=True):

    if k>0:
       if int((tot-k)*prev_time) < 60:
           toappend = str(int((tot-k)*prev_time )) + ' seconds'
       else:
           toappend = str(int((tot-k)*prev_time /60. )) + ' minutes'
       if verbose==True:
           print(str(k) + ' out of ' +str(tot) + ' complete' + ' expected time ' + toappend,end='\r')
    else:
       if verbose==True:
           print(str(k) + ' out of ' +str(tot) + ' complete',end='\r')
    start = time()
    return start

def get_time():
    return time()


def print_expected_time(start_time,count,total_count,every_integer=100,n_workers=1):
    prev_time = get_time()-start_time
    verbose=False
    if int(count) % int(every_integer) ==0:
        verbose=True
    strt = manage_time_print((count+1),total_count,prev_time/n_workers,verbose=verbose)
    return strt