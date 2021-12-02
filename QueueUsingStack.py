class MyQueue(object):

    def __init__(self):
        self.stack = []
        self.index = 0
        self.length = 0

    def push(self, x):
        self.stack.append(x)
        self.length += 1

    def pop(self):
        self.index += 1
        self.length -= 1
        return self.stack[self.index - 1]

    def peek(self):
        return self.stack[self.index]

    def empty(self):
        # return len(self.stack) == self.index
        return self.length == 0

# Your MyQueue object will be instantiated and called as such:
obj = MyQueue()
obj.push(x)
param_2 = obj.pop()
param_3 = obj.peek()
param_4 = obj.empty()