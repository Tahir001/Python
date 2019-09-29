from tkinter import *
import time
import random 

#Create a cavas and name it
tk = Tk()
canvas = Canvas(tk, width=1000, height=600)
tk.title("2D & 3Dimensional Movements")
canvas.pack()

#Creating a ball object to display on the canvas
ball = canvas.create_oval(20,20,60,60, fill="red")
ball2 = canvas.create_oval(20,20,60,60, fill="orange")
xspeed = 1
yspeed = 1

#2 Dimensional Movement
#infinite loop to move the objects continously
x = 1
while x > 0:
    #Horizontal movement, side to side
    for i in range(500):
        canvas.move(ball,1,0)
        canvas.move(ball2,0,1)
        tk.update()
        time.sleep(0.005)
    
    #Vertical Movement, up and down
    for i in range(500):
        canvas.move(ball,-1,0)
        canvas.move(ball2, 0, -1)
        tk.update()
        time.sleep(0.005)
    
    break

''' 3 Dimensional Movement
for i in range(0,1000):
    canvas.create_line(500,300, 200, 200)

#Line = canvas.create_line(500,300,100,100)

'''
tk.mainloop()