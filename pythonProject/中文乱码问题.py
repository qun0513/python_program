'''
1.文件存储为utf-8格式，编码声明为utf-8，# encoding:utf-8
2.出现汉字的地方前面加 u
3.不同编码之间不能直接转换，要经过unicode中间跳转
4.cmd 下不支持utf-8编码
5.raw_input提示字符串只能为gbk编码
'''



''' 
绘图时
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 ,
但一般字体SimHei可能使负号无法显示，可换一个字体'Microsoft YaHei'
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

如果上述依旧不能搞定，那么试试这种形式：

plt.xlabel('name',fontproperties = FontProperties(fname='/System/Library/Fonts/PingFang.ttc'))
plt.ylabel('name',fontproperties = FontProperties(fname='/System/Library/Fonts/PingFang.ttc'))
'''