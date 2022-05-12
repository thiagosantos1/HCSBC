from turtle import *
import turtle
from types import *
import json

TEXT_COLOR = "black"
LINE_COLOR = "blue"

def recursive_read(current_array, data_array):
    if len(data_array[1]) < 1:
        return current_array
    for key in data_array[1].keys():
        print("Adding "+key)
        current_array.append(str(key))
        added_array = recursive_read([],data_array[1][key])
        if len(added_array) > 1:
            current_array.append(added_array)
    return current_array

myTree = ['DECISION']
with open('annotations_structure.json') as json_file:
    data = json.load(json_file)
    myTree.append(recursive_read([], ["",data]))

print(myTree)

s = 50
startpos = (-400,120)
color("white")
def cntstrs(list):
    return len([item for item in list if type(item) is StringType])
def drawtree(tree, pos, head=0,level=0):
    c = cntstrs(tree)
    while len(tree):
        goto(pos)
        item = tree.pop(0)
        if head:
            color(TEXT_COLOR)
            write(item,move=True)
            color(LINE_COLOR)
            drawtree(tree.pop(0),pos,level=2)
        else:
            if type(item) is StringType:
                if(level == 2):
                    newpos = (pos[0] + level * s, pos[1] - cntstrs(tree) * s / 0.5)
                else:
                    newpos = (pos[0] + level * s, pos[1] - cntstrs(tree) * s / 5 - 30)
                down()
                goto((newpos[0] - 15, newpos[1] + 5))
                up()
                goto(newpos)
                color(TEXT_COLOR)
                write(item,move=True)
                color(LINE_COLOR)
                newpos = turtle.pos()
            elif type(item) is ListType:
                drawtree(item,newpos,level=level+1)

up()
drawtree(myTree, startpos,level=1)
hideturtle()
getscreen().getcanvas().postscript(file='annotation_graph.ps')
img = Image.open(io.BytesIO(ps.encode('utf-8')))
img.save('/tmp/test.jpg')
