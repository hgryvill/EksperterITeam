import random as rnd 
from tkinter import * 
import time 
from math import sin, cos, sqrt, exp 
import numpy as np 
 
# Environmental Specifciation 
num = 15  # number of agents 
s = 10  # environment size 
 
# Agent Parametrs (play with these) 
k = 1.5 
m = 2.0 
t0 = 3 
dt = 0.05 
rad = 0.1  # Collision radius 
sight = 10  # Neighbor search range 
maxF = 5  # Maximum force/acceleration 
speed_limit = 2*10 
friend_scale = 10
coffeeScale = 1
pixelsize = 600 
framedelay = 30 
l = 600 
drawVels = True 
number_of_coffeStands = 3
win = Tk() 
canvas = Canvas(win, width=l, height=l, background="#444") 
 
 
walllist = [(0,50,1200,50),(0,550,1200,550),(400,250,800,250),(800,250,850,300),(850,300,800,350),(800,350,400,350), 
            (400,350,350,300),(350,300,400,250)] 
 
walllist = [(0,(2/5)*l,l*(1/5),(2/5)*l),(0,l*(3/5),l*(1/5),l*(3/5)),((2/5)*l,l*(2/5),l*(3/5),l*(2/5)), 
            (l*1/5,(2/5-1/20)*l,l*(2/5),(2/5-1/20)*l),(l*1/5,l*(3/5+1/20),l*(2/5),l*(3/5+1/20)), 
            (l*3/5,(2/5-1/20)*l,l*(4/5),(2/5-1/20)*l),(l*3/5,l*(3/5+1/20),l*(4/5),l*(3/5+1/20)), 
            (l*4/5,(2/5)*l,l,(2/5)*l),(l*4/5,l*(3/5),l,l*(3/5)), 
            (l*1/5, l*2/5, l/5,l*(2/5-1/20)),(l*2/5, l*2/5, l*2/5,l*(2/5-1/20)),(l*3/5, l*2/5, l*3/5,l*(2/5-1/20)),(l*4/5, l*2/5, l*4/5,l*(2/5-1/20)), 
            (l * 1 / 5, l * 3 / 5, l / 5, l * (3 / 5 + 1 / 20)), 
            (l * 2 / 5, l * 3 / 5, l * 2 / 5, l * (3 / 5 + 1 / 20)), 
            (l * 3 / 5, l * 3 / 5, l * 3 / 5, l * (3 / 5 + 1 / 20)), 
            (l * 4 / 5, l * 3 / 5, l * 4 / 5, l * (3 / 5 + 1 / 20))] 
 

 
 
 
#husk type i personklassen 
#husk aa skrive inni updatefunksjonen: 
    #for i in range(len(people)): 
        #update_goal_velocity(people[i]) 
def update_goal_velocity(person): 
    if person.zonetype == 1: 
        update_goal_velocity1(person) 
    elif person.zonetype == 2: 
        update_goal_velocity2(person) 
    elif person.zonetype == 3: 
        update_goal_velocity3(person) 
 
def update_goal_velocity1(person): 
    global s 
    if person.pos[0] > 3*s/5: 
        person.gv[0] = -1 
        person.gv[1] = 0 
        person.gv = np.array([person.gv[0], person.gv[1]]) 
    elif person.pos[0] < 2*s/5: 
        person.gv[0] = 1 
        person.gv[1] = 0 
        person.gv = np.array([person.gv[0], person.gv[1]]) 
    else: 
        person.gv[0] = 0 
        person.gv[1] = 1 
        person.gv = np.array([person.gv[0], person.gv[1]]) 
 
def update_goal_velocity2(person): 
    global s 
    if person.pos[1] > 3*s/5: 
        person.gv[0] = 0 
        person.gv[1] = -1 
        person.gv = np.array([person.gv[0], person.gv[1]]) 
    elif person.pos[1] < 2*s/5: 
        person.gv[0] = 0 
        person.gv[1] = 1 
        person.gv = np.array([person.gv[0], person.gv[1]]) 
    else: 
        person.gv[0] = 1 
        person.gv[1] = 0 
        person.gv = np.array([person.gv[0], person.gv[1]]) 
 
for wall in walllist: 
    line = canvas.create_line(wall) 
walllist = np.array(walllist)/pixelsize*s 
 
 
 
 
canvas.pack() 
 
# Initalized variables 
ittr = 0 
QUIT = False 
paused = False 
step = False 
 
circles = [] 
velLines = [] 
gvLines = [] 
 
deleted_indexes = []
 
class Person: 
    def __init__(self, pos_x = 0, pos_y = 0, vel_x = 0, vel_y = 0, gv_x = 0, gv_y = 0, zonetype=0, friendtype=0, coffeeCharge=0): 
        self.vel = np.array([vel_x, vel_y]) 
        self.pos = np.array([pos_x, pos_y]) 
        self.gv = np.array([gv_x, gv_y]) 
        self.rad = rad 
        self.coffeeCharge = coffeeCharge
        self.neighbors = [] 
        self.nt = [] 
        self.zonetype = zonetype # definerer hvor de vil gaa 
        self.friendtype = friendtype # alle personer med samme friendtype er venner 
 
    def __del__(self): 
        print("People destroyed") 
        
    
people = [None]*num 
 
 
def initSim(): 
    global rad, people 
 
    print("") 
    print("Simulation of Agents on a flat 2D torus.") 
    print("Agents avoid collisions using prinicples based on the laws of anticipation seen in human pedestrians.") 
    print("Agents are white circles, Red agent moves faster.") 
    print("Green Arrow is Goal Velocity, Red Arrow is Current Velocity") 
    print("SPACE to pause, 'S' to step frame-by-frame, 'V' to turn the velocity display on/off.") 
    print("") 
 
    for i in range(num): 
        circles.append(canvas.create_oval(0, 0, rad, rad, fill="white")) 
        velLines.append(canvas.create_line(0, 0, 10, 10, fill="red")) 
        gvLines.append(canvas.create_line(0, 0, 10, 10, fill="green")) 
 
        if i >= np.floor(num/2): 
 
            people[i] = Person(rnd.uniform(8, 10), rnd.uniform(4.15, 5.85), -1, rnd.uniform(-.01, .01), -3, 
                               rnd.uniform(-.01, .01),1,1) 
        else: 
            people[i] = Person(rnd.uniform(0, 2), rnd.uniform(4.15,5.85), 1, rnd.uniform(-.01, .01), 3, 
                               rnd.uniform(-.01, .01),1,1) 
 
 
def drawWorld(): 
    global rad, s
    print("drawWorld people", len(people))
    for i in range(len(people)): 
        scale = pixelsize / s 
        canvas.coords(circles[i], scale * (people[i].pos[0] - rad), scale * (people[i].pos[1] - rad), 
                      scale * (people[i].pos[0] + rad), scale * (people[i].pos[1] + rad)) 
        canvas.coords(velLines[i], scale * people[i].pos[0], scale * people[i].pos[1], 
                      scale * (people[i].pos[0] + 1. * rad * people[i].vel[0]), 
                      scale * (people[i].pos[1] + 1. * rad * people[i].vel[1])) 
        canvas.coords(gvLines[i], scale * people[i].pos[0], scale * people[i].pos[1], 
                      scale * (people[i].pos[0] + 1. * rad * people[i].gv[0]), 
                      scale * (people[i].pos[1] + 1. * rad * people[i].gv[1])) 
        if drawVels: 
            canvas.itemconfigure(velLines[i], state="normal") 
            canvas.itemconfigure(gvLines[i], state="normal") 
        else: 
            canvas.itemconfigure(velLines[i], state="hidden") 
            canvas.itemconfigure(gvLines[i], state="hidden") 
 
 
def findNeighbors(): 
    global people 
 
    for i in range(len(people)): 
        people[i].neighbors = [] 
        people[i].nt = [] 
        vel_angle = np.arctan2(people[i].vel[1], people[i].vel[0]) 
        for j in range(len(people)): 
            if i == j: continue; 
            d = people[i].pos - people[j].pos 
            d_angle = np.arctan2(d[1], d[0]) 
            l2 = d.dot(d) 
            s2 = sight ** 2 
            if l2 < s2 and abs(d_angle-vel_angle)>np.pi/2: 
                people[i].neighbors.append(j) 
                people[i].nt.append(sqrt(l2)) 
 
 
def dE(persona, personb, r): 
    global k, m, t0 
    INFTY = 999 
    maxt = 999 
 
    w = personb.pos - persona.pos 
    v = persona.vel - personb.vel 
    radius = r + r 
    dist = sqrt(w[0] ** 2 + w[1] ** 2) 
    if radius > dist: radius = .99 * dist 
    a = v.dot(v) 
    b = w.dot(v) 
    c = w.dot(w) - radius * radius 
    discr = b * b - a * c 
    if (discr < 0) or (a < 0.001 and a > - 0.001): return np.array([0, 0]) 
    discr = sqrt(discr) 
    t1 = (b - discr) / a 
 
    t = t1 
 
    if (t < 0): return np.array([0, 0]) 
    if (t > maxt): return np.array([0, 0]) 
 
    d = k * exp(-t / t0) * (v - (v * b - w * a) / (discr)) / (a * t ** m) * (m / t + 1 / t0) 
 
    return d 
 
 
def closest_point_line_segment(c, wall): 
    line_start = wall[0:2] 
    line_end = wall[2:4] 
    dota = (c - line_start).dot(line_end - line_start) 
    if dota <= 0: 
        return line_start 
    dotb = (c - line_end).dot(line_start - line_end) 
    if dotb <= 0: 
        return line_end 
    slope = dota / (dota + dotb) 
    return line_start + (line_end - line_start) * slope 
 
 
def normal(wall): 
    # compute normal vector of wall 
    p = wall[2:4] - wall[0:2] 
    norm = np.array([-p[1], p[0]]) 
    return norm/np.sqrt(norm.dot(norm)) 
 
 
def wallforces(person): 
    # wall forces acting on particle with center p and velocity 
    global walllist, rad 
    F = [0, 0] 
 
    for wall in walllist: 
 
        # find closest point to given wall, if too far away, do not care about given wall 
        closest = closest_point_line_segment(person.pos, wall)-person.pos 
        dw = closest.dot(closest) 
        if dw > sight: 
            continue 
 
        r = np.sqrt(dw) if dw < rad**2 else rad 
 
        t_min = 3 
 
        discCollision = 0 
        segmentCollision = 0 
 
        a = person.vel.dot(person.vel) 
 
        # does particle collide with top capsule 
        w_temp = wall[0:2] - person.pos 
        b_temp = w_temp.dot(person.vel) 
        c_temp = w_temp.dot(w_temp) - r ** 2 
        discr_temp = b_temp * b_temp - a * c_temp 
        if discr_temp > 0 and abs(a) > 0: 
            discr_temp = sqrt(discr_temp) 
            t = (b_temp - discr_temp) / a 
            if 0 < t < t_min: 
                t_min = t 
                b = b_temp 
                discr = discr_temp 
                w = w_temp 
                discCollision = 1 
 
        # does particle collide with bottom capsule 
        w_temp = wall[2:4] - person.pos 
        b_temp = w_temp.dot(person.vel) 
        c_temp = w_temp.dot(w_temp) - r ** 2 
        discr_temp = b_temp * b_temp - a * c_temp 
        if discr_temp > 0 and abs(a) > 0: 
            discr_temp = sqrt(discr_temp) 
            t = (b_temp - discr_temp) / a 
            if 0 < t < t_min: 
                t_min = t 
                b = b_temp 
                discr = discr_temp 
                w = w_temp 
                discCollision = 1 
 
        # does particle collide with line segment from the front 
        w1 = wall[0:2] + r * normal(wall) 
        w2 = wall[2:4] + r * normal(wall) 
        w_temp = w2 - w1 
        D = np.cross(person.vel, w_temp) 
        if D != 0: 
            t = np.cross(w_temp, person.pos - w1) / D 
            # s = (p+velocity*t-o1_temp).dot(o_temp)/(o_temp.dot(o_temp)) 
            s = np.cross(person.vel, person.pos - w1) / D 
            if 0 < t < t_min and 0 <= s <= 1: 
                t_min = t 
                w = w_temp 
                discCollision = 0 
                segmentCollision = 1 
 
        # does particle collide with line segment from the bottom 
        w1 = wall[0:2] - r * normal(wall) 
        w2 = wall[2:4] - r * normal(wall) 
        w_temp = w2 - w1 
        D = np.cross(person.vel, w_temp) 
        if D != 0: 
            t = np.cross(w_temp, person.pos - w1) / D 
            # s = (p + velocity * t - o1_temp).dot(o_temp) / (o_temp.dot(o_temp)) 
            s = np.cross(person.vel, person.pos - w1) / D 
            if 0 < t < t_min and 0 <= s <= 1: 
                t_min = t 
                w = w_temp 
                discCollision = 0 
                segmentCollision = 1 
 
        # compute forces acting on the particle 
        if discCollision: 
            FAvoid = -k * np.exp(-t_min / t0) * (person.vel - (b * person.vel - a * w) / discr) / (a * (t_min ** m)) * ( 
                        m / t_min + 1 / t0) 
            mag = np.sqrt(FAvoid.dot(FAvoid)) 
            if (mag > maxF): FAvoid = maxF * FAvoid / mag 
            F += FAvoid 
        if segmentCollision: 
            FAvoid = k * np.exp(-t_min / t0) / (t_min ** m * np.cross(person.vel, w)) * (m / t_min + 1 / t0) * np.array( 
                [-w[1], w[0]]) 
            mag = np.sqrt(FAvoid.dot(FAvoid)) 
            if (mag > maxF): FAvoid = maxF * FAvoid / mag 
            F += FAvoid 
    return F 
 
def Lennard_Jones_gradient(persona, personb,r, scaling_factor): 
    r_vec = personb.pos-persona.pos 
    dist = r_vec.dot(r_vec) 
    unit_vec = r_vec/dist 
    gradient = unit_vec*((-6/(dist-2*r)**7)+(12/(dist-2*r)**13)) 
    gradient_size = np.sqrt(gradient.dot(gradient)) 
    if gradient_size > maxF: 
        gradient = gradient*maxF/gradient_size # scaling down the gradient 
    return gradient/scaling_factor

""" 
def friendForce(): 
    global people 
    for i in range(len(people)): 
        for n,j in enumerate(people[i].neighbors): 
            if people[i].friendtype == j.friendtype and people[i].friendtype!=0: 
                 F[i] 
""" 

def coffeForce(person, scale):
    force = Lenn
    return force


def hardwall(i, dt, a): 
    global people 
 
    collision = False 
    p = people[i].pos + (a*dt)*dt 
 
    r = rad 
 
    for wall in walllist: 
 
        q = closest_point_line_segment(people[i].pos, wall) 
        y = (people[i].pos - q).dot(people[i].pos - q) 
        if y <= rad ** 2: 
            print("oh hell no!") 
 
        q = closest_point_line_segment(p, wall) 
        if (p-q).dot(p-q) <= r**2: 
            collision = True 
            w = wall[2:4]-wall[0:2] 
            n = np.array([-w[1], w[0]]) 
            u = people[i].vel.dot(n)/n.dot(n)*n 
            people[i].vel += -2 * u 
 
    if not collision: 
        people[i].vel += a*dt 
        speed =  np.sqrt(people[i].vel.dot(people[i].vel)) 
        #print("speed2:",speed) 
        if speed > speed_limit: 
            #print("speed:",speed) 
            people[i].vel = people[i].vel*speed_limit/speed 
        people[i].pos += people[i].vel*dt 
 
 
 
def F_hardsphere(): 
    global s, people, pixelsize 
    collisionLastFrame = np.zeros([len(people), len(people)])  # Element i,j er 1 hvis element i og j kolliderte i forrige frame 
    collisionWithWall = np.zeros(len(people))  # person kolliderte med person i forrige frame 
    # Maa sjekke at de ikke kolliderte i forrige frame 
    # print("Avstand:",sqrt(((c[0,0]-c[1,0]) ** 2) + ((c[1,1]-c[0,1]) ** 2))) 
    for i in range(len(people)): 
        for j in range(i + 1, len(people)):  # egentlig: for j in range(i+1, len(people)): 
            d = people[i].pos - people[j].pos 
            # print("distance:", sqrt(d[0]**2+d[1]**2)) 
            if (sqrt((d[0] ** 2) + (d[1] ** 2)) < 2 * rad) and collisionLastFrame[i, j] == False and collisionWithWall[ 
                i] == False:  # Kollisjon 
                collisionLastFrame[i, j] = 1 
                print("d:", d) 
                print("Kollisjon!") 
                if j < len(people):  # kollisjon mellom to personer 
                    d_angle = np.arctan2(d[1], d[0]) 
                    # print(d_angle) 
                    unit_vec = np.array([np.cos(d_angle), np.sin(d_angle)]) 
                    # print("unit vec",unit_vec) 
                    v_i_parallell = np.dot(people[i].vel, unit_vec) * unit_vec 
                    v_j_parallell = np.dot(people[j].vel, unit_vec) * unit_vec 
                    # print("i parallell:", v_i_parallell, "j parallell:", v_j_parallell) 
                    people[i].vel += -v_i_parallell + v_j_parallell 
                    people[j].vel += v_i_parallell - v_j_parallell 
                    # print("v_i:", people[i].vel) 
                    # print("v_j:", v[j]) 
 
            elif (sqrt((d[0] ** 2) + (d[1] ** 2)) < 2 * rad): 
                collisionLastFrame[i, j] = 1 
                # else: 
                #    collisionLastFrame[i,j] = 0 
 

def outside(person,index): 
    print("position for person :", index, ":",person.pos)
    global people 
    if person.pos[1]<0 or person.pos[1]>8 or person.pos[0]<0 or person.pos[0]>10: 
        print("outside")
        #people.remove(person)
        #del person
        #print("lenght people",len(people))
        return True
    return False
 
def update(dt): 
    global people, deleted_indexes
    findNeighbors() 
 
    F = []  # force 
    deleted_indexes = []
    
    for i in range(len(people)): 
        F.append(np.zeros(2)) 
    deleted_indexes = []
    for i in range(len(people)):
        update_goal_velocity(people[i]) 
        F[i] += (people[i].gv - people[i].vel) / .5 
        F[i] += 1 * np.array([rnd.uniform(-1, 1), rnd.uniform(-1, 1)]) 
 
        for n, j in enumerate(people[i].neighbors):  # j is neighboring agent 
            d = people[i].pos - people[j].pos 
            r = rad 
            dist = sqrt(d.dot(d)) 
            if dist < 2 * rad: r = dist / 2.001;  # shrink overlapping agents 
            dEdx = dE(people[i], people[j], r) 
            FAvoid = -dEdx 
            mag = np.sqrt(FAvoid.dot(FAvoid)) 
            if (mag > maxF): FAvoid = maxF * FAvoid / mag 
 
            if people[i].friendtype == people[j].friendtype and people[i].friendtype != 0: 
                F[i] += Lennard_Jones_gradient(people[i],people[j],rad, friend_scale) 
            else: 
                F[i] += FAvoid 
 
 
        FAvoid = wallforces(people[i]) 
        F[i] += FAvoid
        #Fcoffee = CoffeeForce(people[i],coffeeScale)
        #F[i] += Fcoffee
        cond = outside(people[i],i) 
        if cond: deleted_indexes.append(i)
        
    F_hardsphere() 
    for i in range(len(people)): 
        a = F[i] 
        hardwall(i, dt, a) 
 
 
def on_key_press(event): 
    global paused, step, QUIT, drawVels 
    if event.keysym == "space": 
        paused = not paused 
    if event.keysym == "s": 
        step = True 
        paused = False 
    if event.keysym == "v": 
        drawVels = not drawVels 
    if event.keysym == "Escape": 
        QUIT = True 
 
 
def drawFrame(dt=0.05): 
    global start_time, step, paused, ittr, deleted_indexes, circles, velLines, gvLines
 
    if ittr > maxIttr or QUIT:  # Simulation Loop 
        print("%s itterations ran ... quitting" % ittr) 
        win.destroy() 
    else: 
        elapsed_time = time.time() - start_time 
        start_time = time.time() 
        if not paused: 
            # if ittr%100 == 0 : print ittr,"/",maxIttr 
            update(dt) 
            ittr += 1 
            
            # delete some peepz
            deleted_indexes = np.flip(np.sort(deleted_indexes),0)
            print("deleted indexes:", deleted_indexes)
            for i in deleted_indexes:
                print("i:",i)
                canvas.delete(circles[i])
                canvas.delete(velLines[i])
                canvas.delete(gvLines[i])
                people.pop(i)
                circles.pop(i)
                velLines.pop(i)
                gvLines.pop(i)
            print("length people:", len(people))
            if len(people)==0:
                win.destroy()
        drawWorld() 
        if step == True: 
            step = False 
            paused = True 
 
            # win.title("K.S.G. 2014 (Under Review) - " + str(round(1/elapsed_time,1)) +  " FPS") 
        win.title("K.S.G. 2014 (Under Review)") 
        win.after(framedelay, drawFrame) 
 
 
# win.on_resize=resize 
 
win.bind("<space>", on_key_press) 
win.bind("s", on_key_press) 
win.bind("<Escape>", on_key_press) 
win.bind("v", on_key_press) 
 
initSim(); 
maxIttr = 1000 
 
start_time = time.time() 
win.after(framedelay, drawFrame) 
mainloop() 