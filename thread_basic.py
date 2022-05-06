import _thread
import time

def _counter(counter, thread_name):
  while (counter):
    time.sleep(0.01)
    print("{}: {}".format(thread_name, counter))
    counter -= 1

counter = 5

# Khởi tạo 2 threads 1 và 2
try:
  _thread.start_new_thread(_counter, (counter, "khanh thread")) # pass counter and thread_name into method _counter
  print("\n")
  _thread.start_new_thread(_counter, (counter, "ai thread"))
except:
  print("Error: unable to start thread")

# Running counter
while (counter):
  counter -= 1
  pass