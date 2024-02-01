class A():
    def printme(self):
        print("hi")
    def __str__(self):
        return self.__class__.__str__()

a = "A"
print(str(a))

# x = [
#     {"si":"SI","cn":"cn1"},
#     {"si":"SI2","cn":"cn12"}
# ]

# x = str(x)
# x = x.replace(",",";")
# print(x)
#
# for a in x:
#     print(a["si"])