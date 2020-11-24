#Keep track of time
import time

def checkRuntime(message, old_time):
    new_time = time.time()
    print(message, ':', new_time - old_time)
    return new_time