from tkinter import Tk, Canvas, PhotoImage, mainloop
from itertools import product
from numpy import argmax
from time import sleep

from MS import World


# Init Tkinter canvas
WIDTH, HEIGHT = 64, 64
window = Tk()
canvas = Canvas(window, width=WIDTH * 10, height=HEIGHT * 10, bg="#000000")
canvas.pack()

# Load map
world = World("Caldera")
probabilities, real_goals = list(), list()

# Predict for each observation
for i in range(len(world.observations)):
    # Compute goal probability
    probs = world.predictMastersSardina(world.observations[i])
    pred_goal = argmax(probs)
    
    probabilities.append(probs)
    real_goals.append(world.goals[argmax(world.labels[i])])
    
    window.title(f"Goal probability: {max(probs)}")

    # Generate map image
    img = PhotoImage(width=WIDTH, height=HEIGHT)

    for x, y in product(range(WIDTH), range(HEIGHT)):
        if world.grid[x][y] is not None:
            img.put("#FFFFFF", (x, y))
    
    # Display start and current positions
    img.put("#FF2600", tuple(world.observations[i].point))
    img.put("#FF9300", tuple(world.start.point))
    
    # Display goals
    for goal in world.goals:
        img.put("#0096FF", tuple(goal.point))
    
    # Display path to goal
    path_s_g = world.planner(world.start, world.observations[i])
    path_p_g = world.planner(world.observations[i], world.goals[pred_goal])

    for node in path_s_g[1:-1]:
        img.put("#9EC591", tuple(node.point))
    for node in path_p_g[1:-1]:
        img.put("#008F00", tuple(node.point))
    
    # Create image
    img = img.zoom(10, 10)
    canvas.create_image((WIDTH * 5, HEIGHT * 5), image=img, state="normal")

    window.update()
    sleep(.5)

accuracy = world.accuracy(probabilities, real_goals)
print(f"Global accuracy for this map: {accuracy}%")
