#控制结构（顺序、分支、循环）  分支结构  循环结构  文件操作
'''guess=eval(input())
if guess==9:
    print ('猜对了')
    if True:
        print('中奖了')'''
#程序的异常处理
'''try :
    a=1
except(NameError):
    b=1'''#else finally#
'''for a in 'python123':
    print(a,end=',')'''
#遍历循环：计数、字符串、列表、文件
f=open('D:/PY/3.output.txt')
'''read() 每次读取整个文件，它通常将读取到底文件内容放到一个字符串变量中，也就是说 .read() 生成文件内容是一个字符串类型。
readline()每只读取文件的一行，通常也是读取到的一行内容放到一个字符串变量中，返回str类型。
readlines()每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型。'''

a=f.read()
b=f.readline()
c=f.readlines()
print(c)
#print(d)
#print(f)
#print(a)
#读取指定行
#import linecache
'''d=open('D:/PY/3.rain.txt')
with open(d,'r') as file:
    line=file.readline()
    counts=1
    while line:
        if counts>=10:
            break
        line=file.readline()
        counts+=1'''
