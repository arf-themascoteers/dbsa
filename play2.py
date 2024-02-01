# class A():
#     def printme(self):
#         print("hi")
#
#
# a = "A"
#
# a.printme()


x = [
    {"si":"SI","cn":"cn1"},
    {"si":"SI2","cn":"cn12"}
]

x = str(x)
x = x.replace(",",";")
print(x)