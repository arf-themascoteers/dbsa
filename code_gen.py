import copy

# index = 4
#
# for i in range(1,4):
#     for j in range(1,25):
#         print(f"alpha{i}_{j} = params[{index}]")
#         index = index + 1

# counter = 1
#
#
# def gen(waiting, inqueue):
#     global counter
#     if len(waiting) == 0:
#         x = ",".join(inqueue)
#         x = f"{x},alpha1_{counter},alpha2_{counter},alpha3_{counter}"
#         x = "gndis.append(self.out("+x+"))"
#         print(x)
#         counter = counter+1
#     else:
#         for i in range(len(waiting)):
#             now_waiting = waiting[0:i]+waiting[i+1:]
#             inqueue_now = inqueue + [waiting[i]]
#             gen(now_waiting, inqueue_now)
#
#
# gen(["i","j","k","l"],[])


for i in range(1,4):
    for j in range(1,25):
        print(f"\"alpha{i}_{j}\",")