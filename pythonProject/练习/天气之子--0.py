#--------------------1--------------------------
'''
bookshelf = [
  "The Effective Engineer",
  "The 4 hours work week",
  "Zero to One",
  "Lean Startup",
  "Hooked"
]
for book in bookshelf:
    print(book)
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
dictionary_tk = {
  "name": "Leandro",
  "nickname": "Tk",
  "nationality": "Brazilian",
  "age": 24
}

for attribute, value in dictionary_tk.items():
    print("My %s is %s" %(attribute, value))

class Vehicle_01:
    def __init__(self, number_of_wheels, type_of_tank, seating_capacity, maximum_velocity):
        self.number_of_wheels = number_of_wheels
        self.type_of_tank = type_of_tank
        self.seating_capacity = seating_capacity
        self.maximum_velocity = maximum_velocity
    def number_of_wheels(self):
        return self.number_of_wheels

    def set_number_of_wheels(self, number):
        self.number_of_wheels = number
tesla_model_s = Vehicle_01(4, 'electric', 5, 250)
print(tesla_model_s.type_of_tank) # 4
tesla_model_s.number_of_wheels = 2 # setting number of wheels to 2
print(tesla_model_s.number_of_wheels) # 2
'''

#-----------------------2----------------------------

 #怎么快速打印出包含所有 ASCII 字母（大写和小写）的字符串
import string
a=string.ascii_letters
print(a)
b='我喜欢听周杰伦的歌'
b1='i like to listen to music.'
c=b.center(24,'-')
print(c)
d=b.find('周杰伦') #用 find 方法，如果找到，就返回子串的第一个字符的索引，否则返回 -1
print(d)
print(b1.title(),'\n',string.capwords(b1))#string.capwords()

l=[0,1,2,3,4,1]
print(l)
print(l.clear())
k = [1, 2, 3]
k[::] = []
print(k)
print(l.count(1))

m=[1,2,3]
n=[4,5,6]
m.extend(n)
print(m)
print('\n',m+n)

m.insert(2,'iii')
print(m)
m.pop(2)
print(m)
m.remove(1)
print(m)
m.reverse()
print(m)
m=m[::-1]
print(m)

p=(1,)  #元组
print(type(p))

r=' i love you'
r=r.replace('love','hate')
print(r)
r=r.strip()
print(r)
r=r.split()
print(r)

import webbrowser
#webbrowser.open('http://www.python.org')

import random
s=random.choice([1,2,3,4])
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
plt.plot([1,2,3],[4,6.5,8],c='r',lw=1,ls='--')
ax.text(2,5,s)
plt.show()
