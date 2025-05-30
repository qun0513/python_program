#2021.08.23
#函数 变量作用域 模块 包
def add(a,b):#形参，形参可以取默认值，在参数列表的后面位置
    return a+b
d=3#实参
e=7
c=add(d,e)#实参
print(c)
f=(1,2)
g=(3,4)
print(add(f,g))#元组合并 字符串连接 列表合并---参数的多态性
#如果要避免列表在函数中被修改，可使用列表的拷贝作为实参
#global作用域用于在函数内部声明全局变量
#nonlocal用于闭包（函数中的函数）
a=1
def test():
    a=10
    def show():
        nonlocal a
        a=100
        print('a in show',a)
    show()
    print('a in test',a)
test()
print('a in main',a)
