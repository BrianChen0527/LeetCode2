class MyStack(object):

    def __init__(self):
        self.queue = []
        self.length = 0

    def push(self, x):
        self.queue.append(x)
        self.length += 1

    def pop(self):
        popped = self.top()
        self.queue.pop()
        self.length -= 1
        return popped

    def top(self):
        return self.queue[self.length - 1]

    def empty(self):
        return self.length == 0

obj = MyStack()
obj.push("sdfs")
param_2 = obj.pop()
param_3 = obj.top()
param_4 = obj.empty()