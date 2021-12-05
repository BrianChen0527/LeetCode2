import collections


class MyStack(object):

    def __init__(self):
        self.queue = collections.deque()

    def push(self, x):
        self.queue.append(x)
        for i in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    def pop(self):
        return self.queue.popleft()

    def top(self):
        return self.queue[0]

    def empty(self):
        return len(self.queue) == 0

obj = MyStack()
obj.push("dsf")
param_2 = obj.pop()
param_3 = obj.top()
param_4 = obj.empty()