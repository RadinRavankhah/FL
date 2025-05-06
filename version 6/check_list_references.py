class a:
    def __init__(self, l):
        self.list = l

l = [1, 2, 3, 4]
obj = a(l)

obj.list[0] = 7
print(l)

# [1, 2, 3, 4]
# turned to
# [7, 2, 3, 4]