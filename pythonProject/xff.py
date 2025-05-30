#!/usr/bin/env python
# coding: utf-8

# 绘制玫瑰花并添加文字
import turtle

# 设置画布大小
# turtle.screensize(canvwidth=None, canvheight=None, bg=None)
turtle.setup(width=0.6, height=0.6)
# 设置初始位置
turtle.penup()
turtle.left(90)
turtle.fd(200)
turtle.pendown()
turtle.right(90)

# 输出文字
printer = turtle.Turtle()
printer.hideturtle()
printer.penup()
printer.back(400)
printer.write("to : 小菲菲\n", align="left", font=("楷体", 18, "bold"))
printer.write("from : ZQ\n\n2022.12.26--forever", align="left", font=("楷体", 18, "normal"))
#printer.write("\n2022.12.26--forever", align="left", font=("楷体", 18, "normal"))

# 花蕊
turtle.fillcolor("red")
turtle.begin_fill()
turtle.circle(10, 180)
turtle.circle(25, 110)
turtle.left(50)
turtle.circle(60, 45)
turtle.circle(20, 170)
turtle.right(24)
turtle.fd(30)
turtle.left(10)
turtle.circle(30, 110)
turtle.fd(20)
turtle.left(40)
turtle.circle(90, 70)
turtle.circle(30, 150)
turtle.right(30)
turtle.fd(15)
turtle.circle(80, 90)
turtle.left(15)
turtle.fd(45)
turtle.right(165)
turtle.fd(20)
turtle.left(155)
turtle.circle(150, 80)
turtle.left(50)
turtle.circle(150, 90)
turtle.end_fill()

# 花瓣1
turtle.left(150)
turtle.circle(-90, 70)
turtle.left(20)
turtle.circle(75, 105)
turtle.setheading(60)
turtle.circle(80, 98)
turtle.circle(-90, 40)

# 花瓣2
turtle.left(180)
turtle.circle(90, 40)
turtle.circle(-80, 98)
turtle.setheading(-83)

# 叶子1
turtle.fd(30)
turtle.left(90)
turtle.fd(25)
turtle.left(45)
turtle.fillcolor("green")
turtle.begin_fill()
turtle.circle(-80, 90)
turtle.right(90)
turtle.circle(-80, 90)
turtle.end_fill()
turtle.right(135)
turtle.fd(60)
turtle.left(180)
turtle.fd(85)
turtle.left(90)
turtle.fd(80)

# 叶子2
turtle.right(90)
turtle.right(45)
turtle.fillcolor("green")
turtle.begin_fill()
turtle.circle(80, 90)
turtle.left(90)
turtle.circle(80, 90)
turtle.end_fill()
turtle.left(135)
turtle.fd(60)
turtle.left(180)
turtle.fd(60)
turtle.right(90)
turtle.circle(200, 60)

turtle.done()
turtle.mainloop()


'''
import turtle

turtle.bgcolor("black")
turtle.pensize(2)
sizeh = 1.2


def curve():
    for ii in range(200):
        turtle.right(1)
        turtle.forward(1 * sizeh)


turtle.speed(0)
turtle.color("red", "red")
turtle.begin_fill()
turtle.left(140)
turtle.forward(111.65 * sizeh)
curve()
turtle.left(120)
curve()
turtle.forward(111.65 * sizeh)
turtle.end_fill()
turtle.hideturtle()
'''
