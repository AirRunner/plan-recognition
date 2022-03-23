from tkinter import StringVar, Tk, Canvas, PhotoImage, font, Label, LEFT, CENTER, mainloop
from itertools import product
from numpy import argmax
from time import sleep

from MS import World


# Init Tkinter canvas
WIDTH, HEIGHT = 64, 64
window = Tk()
window.title("Reconnaissance de plan symbolique")
custom_font = font.Font(family='Avenir', size=30, weight="bold")

canvas = Canvas(window, width=WIDTH * 10, height=HEIGHT * 10, bg="#000000")
canvas.pack()

iter_text = StringVar()
info_text = StringVar()
info_text.set("Carte:\nCoût:\nProbabilité du but:")
label = Label(window, textvariable=info_text, font=("SF Mono", 27), justify=LEFT)
label.pack()

# Load map
map = "Caldera"
world = World(map)
probabilities, real_goals = list(), list()

# Predict for each observation
for i in range(len(world.observations)):
    iter_text.set(str(i))
    
    # Compute goal probability
    probs = world.predictMastersSardina(world.observations[i])
    pred_goal = argmax(probs)
    
    probabilities.append(probs)
    real_goals.append(world.goals[argmax(world.labels[i])])

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
    path_s_p = world.planner(world.start, world.observations[i])
    path_s_g = world.planner(world.start, world.goals[pred_goal])
    path_p_g = world.planner(world.observations[i], world.goals[pred_goal])

    for node in path_s_p[1:-1]:
        img.put("#9EC591", tuple(node.point))
    for node in path_p_g[1:-1]:
        img.put("#008F00", tuple(node.point))
    
    # Display general informations
    s_p = "Départ → Position (coût G)"
    p_g = "Position → But (coût H)"
    s_g = "Départ → But"
    prob = "Probabilité du but"
    soleil = "soleil"
    max_w = len(s_p) + 1

    info_text.set(f"\n——  Carte {map}  ——\n\
{s_p:{max_w}} {len(path_s_p): >{5}}\n\
{p_g:{max_w}} {len(path_p_g): >{5}}\n\
{s_g:{max_w}} {len(path_s_g): >{5}}\n\
{prob:{max_w}} {str(max(probs))[:4]: >{5}}\
\n")

    # Create image and text
    img = img.zoom(10, 10)
    canvas.delete("all")
    canvas.create_image((WIDTH * 5, HEIGHT * 5), image=img, state="normal")
    canvas.create_rectangle(10, 10, 90, 50, fill='#424242', width=0)
    canvas.create_text((50, 31), text=str(i + 1), fill="#FF9800", font=custom_font)

    window.update()
    sleep(.5)

accuracy = world.accuracy(probabilities, real_goals)
print(f"Global accuracy for this map: {accuracy}%")

canvas.create_rectangle(140, 220, 500, 420, fill='#424242', width=0)
canvas.create_text(
    (320, 320), text=f"Justesse globale\npour cette carte :\n\n{accuracy}% !",
    fill="#FF9800", font=custom_font, justify=CENTER
)

mainloop()
