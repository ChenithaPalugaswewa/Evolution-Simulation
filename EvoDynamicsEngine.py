import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from matplotlib.patches import Polygon
from matplotlib.offsetbox import AnchoredText

# =========================
# PARAMETERS
# =========================
GRID_SIZE = 50
NUM_CREATURES = 30
INITIAL_FOOD = 60
FOOD_MAX = 150
FOOD_GROWTH_RATE = 0.35
FOOD_DECAY_RATE = 0.01
MUTATION_RATE = 0.08
MIN_POPULATION = 6

SPEED_RANGE = (0.6, 3.2)
VISION_RANGE = (3, 12)
SIZE_RANGE = (0.8, 3.0)
EFFICIENCY_RANGE = (0.6, 1.6)

TERRITORY_SIZE = 10
SEASON_LENGTH = 200
DISASTER_CHANCE = 0.004

PACK_RADIUS = 5
YEARS_PER_STEP = 10000

TRAIL_LENGTH = 12
WANDER_STRENGTH = 0.6
SEPARATION_RADIUS = 1.2
SEPARATION_STRENGTH = 0.6
COHESION_STRENGTH = 0.03
ALIGNMENT_STRENGTH = 0.05

MAX_CREATURES = 500
MAX_HEADING_PATCHES = MAX_CREATURES
MAX_TRAIL_LINES = MAX_CREATURES

species_colors = {}
paused = False
current_step = 0
current_season = 0
season_multiplier = 1.0
total_years = 0.0

# =========================
# UTILITIES
# =========================
def clamp(v, a, b):
    return max(a, min(b, v))

def format_years(total_years):
    if total_years >= 1_000_000_000:
        return f"{total_years/1_000_000_000:.2f} billion years"
    elif total_years >= 1_000_000:
        return f"{total_years/1_000_000:.2f} million years"
    elif total_years >= 1_000:
        return f"{total_years/1_000:.2f} thousand years"
    else:
        return f"{total_years:.0f} years"

# =========================
# EVENT LOG
# =========================
MAX_LOG_LINES = 6
event_log_lines = []

def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    event_log_lines.append(f"[{timestamp}] {message}")
    if len(event_log_lines) > MAX_LOG_LINES:
        event_log_lines.pop(0)

def update_event_box():
    text = "\n".join(event_log_lines)
    if hasattr(update_event_box, "box"):
        update_event_box.box.txt.set_text(text)
    else:
        update_event_box.box = AnchoredText(
            text, loc='lower center', prop=dict(size=8), frameon=True
        )
        update_event_box.box.patch.set_alpha(0.3)
        ax.add_artist(update_event_box.box)

# =========================
# CREATURE CLASS
# =========================
class Creature:
    id_counter = 0
    species_counter = 1

    def __init__(self, x, y, speed=None, vision=None, size=None, efficiency=None, species_id=None, specialization=None):
        self.id = Creature.id_counter
        Creature.id_counter += 1
        self.x = float(x)
        self.y = float(y)
        self.energy = 30.0

        base_speed = random.uniform(*SPEED_RANGE)
        self.max_speed = speed if speed is not None else clamp(base_speed, SPEED_RANGE[0], SPEED_RANGE[1])
        self.vision = vision if vision is not None else random.randint(*VISION_RANGE)
        self.body_size = size if size is not None else random.uniform(*SIZE_RANGE)
        self.efficiency = efficiency if efficiency is not None else random.uniform(*EFFICIENCY_RANGE)
        self.specialization = specialization if specialization else random.choice(['predator','forager','scavenger'])

        if species_id is None:
            self.species_id = f"{Creature.species_counter:03d}"
            Creature.species_counter += 1
            species_colors[self.species_id] = np.clip(np.random.rand(3,), 0.15, 0.95)
            log_event(f"New species {self.species_id} appeared: speed={self.max_speed:.2f}, vision={self.vision}, size={self.body_size:.2f}, eff={self.efficiency:.2f}, spec={self.specialization}")
        else:
            self.species_id = species_id

        angle = random.random()*2*np.pi
        self.vx = np.cos(angle) * (self.max_speed*0.5)
        self.vy = np.sin(angle) * (self.max_speed*0.5)
        self.orientation = angle
        self.trail = [(self.x, self.y)] * TRAIL_LENGTH
        self.wander_angle = random.uniform(-1,1)

    def trait_score(self):
        score = self.max_speed*0.4 + self.vision*0.3 + self.body_size*0.15 + self.efficiency*1.5
        if self.specialization == 'predator':
            score *= 1.12
        elif self.specialization == 'scavenger':
            score *= 0.92
        return score

    def steer_towards(self, tx, ty, strength=1.0):
        dx = tx - self.x
        dy = ty - self.y
        if dx > GRID_SIZE/2: dx -= GRID_SIZE
        if dx < -GRID_SIZE/2: dx += GRID_SIZE
        if dy > GRID_SIZE/2: dy -= GRID_SIZE
        if dy < -GRID_SIZE/2: dy += GRID_SIZE
        dist = np.hypot(dx, dy) + 1e-8
        desired_vx = (dx/dist) * self.max_speed
        desired_vy = (dy/dist) * self.max_speed
        steer_x = (desired_vx - self.vx) * strength
        steer_y = (desired_vy - self.vy) * strength
        return steer_x, steer_y

    def flee_from(self, fx, fy, strength=1.0):
        sx, sy = self.steer_towards(fx, fy, strength=1.0)
        return -sx*strength, -sy*strength

    def wander(self):
        self.wander_angle += random.uniform(-0.5, 0.5) * WANDER_STRENGTH
        angle = self.wander_angle
        tx = self.x + np.cos(angle) * (2 + self.max_speed*2)
        ty = self.y + np.sin(angle) * (2 + self.max_speed*2)
        return self.steer_towards(tx, ty, strength=0.3)

    def separation(self, neighbors):
        sx = sy = 0.0
        for n in neighbors:
            dx = n.x - self.x
            dy = n.y - self.y
            if dx > GRID_SIZE/2: dx -= GRID_SIZE
            if dx < -GRID_SIZE/2: dx += GRID_SIZE
            if dy > GRID_SIZE/2: dy -= GRID_SIZE
            if dy < -GRID_SIZE/2: dy += GRID_SIZE
            d = np.hypot(dx, dy)
            if 0 < d < SEPARATION_RADIUS:
                sx -= (dx / (d+1e-8)) * (SEPARATION_STRENGTH / (d+1e-6))
                sy -= (dy / (d+1e-8)) * (SEPARATION_STRENGTH / (d+1e-6))
        return sx, sy

    def cohesion_and_alignment(self, neighbors):
        if not neighbors:
            return 0.0, 0.0, 0.0, 0.0
        avg_x = np.mean([n.x for n in neighbors])
        avg_y = np.mean([n.y for n in neighbors])
        avg_vx = np.mean([n.vx for n in neighbors])
        avg_vy = np.mean([n.vy for n in neighbors])
        sx, sy = self.steer_towards(avg_x, avg_y, strength=COHESION_STRENGTH)
        ax = (avg_vx - self.vx) * ALIGNMENT_STRENGTH
        ay = (avg_vy - self.vy) * ALIGNMENT_STRENGTH
        return sx, sy, ax, ay

    # --------------------------
    # MOVE WITH SMOOTH VELOCITY & TRAIL INTERPOLATION
    # --------------------------
    def move(self, creatures, food_list):
        neighbors = [n for n in creatures if n.id != self.id and np.hypot(n.x-self.x,n.y-self.y)<self.vision]
        predators = [o for o in neighbors if o.trait_score() > self.trait_score() and np.hypot(o.x-self.x,o.y-self.y)<self.vision*0.9]
        steer_x = steer_y = 0.0

        # Flee predators
        if predators:
            px = np.mean([p.x for p in predators])
            py = np.mean([p.y for p in predators])
            sx, sy = self.flee_from(px, py, strength=1.5)
            steer_x += sx
            steer_y += sy
            log_event(f"Creature {self.id} evaded predators nearby")
        else:
            visible_food = sorted(food_list, key=lambda f: (f[0]-self.x)**2 + (f[1]-self.y)**2)
            if visible_food:
                closest_food = visible_food[0]
                sx, sy = self.steer_towards(closest_food[0], closest_food[1], strength=1.0)
                steer_x += sx
                steer_y += sy
            else:
                sx, sy = self.wander()
                steer_x += sx
                steer_y += sy

        if self.specialization == 'predator' and neighbors:
            prey_candidates = [n for n in neighbors if n.trait_score() < self.trait_score()]
            if prey_candidates:
                target = min(prey_candidates, key=lambda p: (p.x-self.x)**2 + (p.y-self.y)**2)
                sx, sy = self.steer_towards(target.x, target.y, strength=0.6)
                steer_x += sx
                steer_y += sy

        sep_x, sep_y = self.separation(neighbors)
        coh_x, coh_y, align_x, align_y = self.cohesion_and_alignment([n for n in neighbors if n.species_id==self.species_id])
        steer_x += sep_x + coh_x + align_x
        steer_y += sep_y + coh_y + align_y

        # Smooth velocity
        smooth_factor = 0.13
        self.vx = self.vx*(1-smooth_factor) + steer_x*smooth_factor
        self.vy = self.vy*(1-smooth_factor) + steer_y*smooth_factor

        speed = np.hypot(self.vx, self.vy) + 1e-8
        if speed > self.max_speed:
            self.vx = (self.vx/speed)*self.max_speed
            self.vy = (self.vy/speed)*self.max_speed

        # Wrap detection
        new_x = self.x + self.vx
        new_y = self.y + self.vy
        wrapped = new_x<0 or new_x>=GRID_SIZE or new_y<0 or new_y>=GRID_SIZE
        self.x = new_x % GRID_SIZE
        self.y = new_y % GRID_SIZE

        # Interpolated trail
        if wrapped:
            self.trail = [(self.x,self.y)]*TRAIL_LENGTH
        else:
            prev_x, prev_y = self.trail[-1]
            steps = int(max(1,np.hypot(self.vx,self.vy)))
            for i in range(1,steps+1):
                inter_point = (prev_x + self.vx*i/steps, prev_y + self.vy*i/steps)
                self.trail.append(inter_point)
            self.trail = self.trail[-TRAIL_LENGTH:]

        self.orientation = np.arctan2(self.vy,self.vx)
        self.energy -= 0.06 * (np.hypot(self.vx,self.vy)/max(0.5,self.max_speed)) * (self.body_size/self.efficiency + 0.5)

# --------------------------
# REPRODUCTION
# --------------------------
    def reproduce(self):
        new_speed = clamp(self.max_speed + random.uniform(-MUTATION_RATE,MUTATION_RATE), SPEED_RANGE[0], SPEED_RANGE[1])
        new_vision = int(clamp(self.vision + random.choice([-1,0,1]), VISION_RANGE[0], VISION_RANGE[1]))
        new_size = clamp(self.body_size + random.uniform(-MUTATION_RATE*0.4, MUTATION_RATE*0.4), SIZE_RANGE[0], SIZE_RANGE[1])
        new_eff = clamp(self.efficiency + random.uniform(-MUTATION_RATE*0.15, MUTATION_RATE*0.15), EFFICIENCY_RANGE[0], EFFICIENCY_RANGE[1])
        new_spec = self.specialization if random.random()>0.02 else random.choice(['predator','forager','scavenger'])
        species_id = None
        if abs(new_speed-self.max_speed)>0.5 or abs(new_size-self.body_size)>0.5 or abs(new_eff-self.efficiency)>0.2:
            species_id = f"{Creature.species_counter:03d}"
            Creature.species_counter += 1
            species_colors[species_id] = np.clip(np.random.rand(3,), 0.15, 0.95)
            log_event(f"Year {format_years(total_years)}: New species {species_id} appeared")
        offspring = Creature(self.x,self.y,new_speed,new_vision,new_size,new_eff,species_id,new_spec)
        self.energy -= 16
        return offspring

# --------------------------
# SIMULATION SETUP
# --------------------------
def reset_simulation():
    global creatures, food, species_colors, current_step, total_years
    species_colors.clear()
    Creature.id_counter = 0
    Creature.species_counter = 1
    creatures[:] = [Creature(random.uniform(0,GRID_SIZE), random.uniform(0,GRID_SIZE)) for _ in range(NUM_CREATURES)]
    food[:] = [(random.randint(0,GRID_SIZE-1), random.randint(0,GRID_SIZE-1)) for _ in range(INITIAL_FOOD)]
    current_step = 0
    total_years = 0.0
    log_event("Simulation reset!")

creatures = []
food = []
reset_simulation()

# --------------------------
# SIMULATION STEP FUNCTIONS
# --------------------------
def get_territory(x,y):
    return int(x//TERRITORY_SIZE), int(y//TERRITORY_SIZE)

def update_season():
    global season_multiplier, current_season
    current_season += 1
    if current_season % 2==0:
        season_multiplier = 0.7
        log_event(f"Year {format_years(total_years)}: Harsh season")
    else:
        season_multiplier = 1.3
        log_event(f"Year {format_years(total_years)}: Abundant season")

def step():
    global creatures, food, current_step, total_years
    current_step += 1
    total_years += YEARS_PER_STEP

    if len(creatures) < MIN_POPULATION:
        reset_simulation()
        return

    if current_step % SEASON_LENGTH == 0:
        update_season()

    if random.random() < DISASTER_CHANCE:
        affected = random.sample(creatures,k=max(1,len(creatures)//6))
        for c in affected:
            c.energy *= 0.5
        log_event(f"Disaster affected {len(affected)} creatures")

    for c in creatures:
        c.move(creatures, food)

    # Food consumption
    for c in creatures:
        for f in food[:]:
            if int(c.x)==f[0] and int(c.y)==f[1]:
                energy_gain = 14 * (c.efficiency/c.body_size)*season_multiplier
                c.energy += energy_gain
                try: food.remove(f)
                except ValueError: pass
                break

    # Predator hunting
    for c in creatures:
        if c.specialization=='predator':
            nearby = [o for o in creatures if o.id!=c.id and np.hypot(o.x-c.x,o.y-c.y)<(c.vision*0.35)]
            if nearby:
                target = min(nearby,key=lambda p: p.trait_score())
                prob_win = c.trait_score()/(c.trait_score()+target.trait_score()+1e-8)
                if random.random() < prob_win:
                    energy_gain = 0.6*target.energy
                    c.energy += energy_gain
                    target.energy = 0
                    log_event(f"Predator {c.id} hunted {target.id}")

    # Reproduction
    new_creatures=[]
    for c in creatures:
        if c.energy>=50 and len(creatures)+len(new_creatures)<MAX_CREATURES:
            new_creatures.append(c.reproduce())
    creatures.extend(new_creatures)

    # Remove dead
    creatures[:] = [c for c in creatures if c.energy>0]

    # Food regeneration & decay
    territory_counts = {}
    for f in food:
        key = get_territory(f[0],f[1])
        territory_counts[key] = territory_counts.get(key,0)+1
    for tx in range(GRID_SIZE//TERRITORY_SIZE):
        for ty in range(GRID_SIZE//TERRITORY_SIZE):
            key=(tx,ty)
            current=territory_counts.get(key,0)
            max_food = FOOD_MAX//((GRID_SIZE//TERRITORY_SIZE)**2)
            if current<max_food and random.random()<FOOD_GROWTH_RATE*season_multiplier:
                fx = random.randint(tx*TERRITORY_SIZE,(tx+1)*TERRITORY_SIZE-1)
                fy = random.randint(ty*TERRITORY_SIZE,(ty+1)*TERRITORY_SIZE-1)
                food.append((fx,fy))
    for f in food[:]:
        if random.random()<FOOD_DECAY_RATE:
            try: food.remove(f)
            except ValueError: pass

# --------------------------
# CLICK PAUSE
# --------------------------
def on_click(event):
    global paused
    paused = not paused

# --------------------------
# VISUALIZATION
# --------------------------
fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(121)
stats_ax = fig.add_subplot(122)
fig.canvas.mpl_connect('button_press_event',on_click)

ax.set_xlim(0,GRID_SIZE)
ax.set_ylim(0,GRID_SIZE)
ax.set_aspect('equal',adjustable='box')
ax.set_title("Ecosystem — click to pause/resume")

scat = ax.scatter([],[],s=[],c=[],edgecolors='k',linewidths=0.4,zorder=3)

heading_patches=[]
for _ in range(MAX_HEADING_PATCHES):
    poly=Polygon([[0,0],[0,0],[0,0]],closed=True,facecolor=(0.5,0.5,0.5),edgecolor='k',visible=False,zorder=4)
    ax.add_patch(poly)
    heading_patches.append(poly)

trail_lines=[]
for _ in range(MAX_TRAIL_LINES):
    line,=ax.plot([],[],lw=1.0,alpha=0.35,zorder=2)
    trail_lines.append(line)

food_scat=ax.scatter([],[],c='green',marker='*',s=60,zorder=1)

speed_vals,vision_vals,size_vals,pop_vals,years_vals=[],[],[],[],[]

def triangle_vertices(x,y,orientation,scale=1.0):
    nose=(x+np.cos(orientation)*0.8*scale,y+np.sin(orientation)*0.8*scale)
    left_angle=orientation+np.pi*0.7
    right_angle=orientation-np.pi*0.7
    left=(x+np.cos(left_angle)*0.45*scale,y+np.sin(left_angle)*0.45*scale)
    right=(x+np.cos(right_angle)*0.45*scale,y+np.sin(right_angle)*0.45*scale)
    return [nose,left,right]

def update_artists():
    xs=[c.x for c in creatures]
    ys=[c.y for c in creatures]
    sizes=[max(6,c.body_size*36) for c in creatures]
    colors=[species_colors[c.species_id] for c in creatures]
    scat.set_offsets(np.c_[xs,ys]) if creatures else scat.set_offsets(np.empty((0,2)))
    scat.set_sizes(sizes)
    scat.set_color(colors)

    for i,poly in enumerate(heading_patches):
        if i<len(creatures):
            c=creatures[i]
            verts=triangle_vertices(c.x,c.y,c.orientation,scale=(0.8+c.body_size*0.18))
            poly.set_xy(verts)
            poly.set_facecolor(species_colors[c.species_id])
            poly.set_edgecolor('k')
            poly.set_alpha(0.95)
            poly.set_visible(True)
        else:
            poly.set_visible(False)

    for i,line in enumerate(trail_lines):
        if i<len(creatures):
            trail=creatures[i].trail
            xs_t=[p[0] for p in trail]
            ys_t=[p[1] for p in trail]
            line.set_data(xs_t,ys_t)
            col=species_colors[creatures[i].species_id]
            line.set_color(col)
            line.set_alpha(0.25+0.05*creatures[i].body_size)
            line.set_linewidth(1.0+creatures[i].body_size*0.4)
            line.set_visible(True)
        else:
            line.set_visible(False)

    if food:
        fx=[f[0]+0.0 for f in food]
        fy=[f[1]+0.0 for f in food]
        food_scat.set_offsets(np.c_[fx,fy])
    else:
        food_scat.set_offsets(np.empty((0,2)))

    update_event_box()

def update(frame_num):
    if not paused:
        step()
    update_artists()

    if creatures:
        speed_vals.append(np.mean([c.max_speed for c in creatures]))
        vision_vals.append(np.mean([c.vision for c in creatures]))
        size_vals.append(np.mean([c.body_size for c in creatures]))
        pop_vals.append(len(creatures))
        years_vals.append(total_years)

    stats_ax.clear()
    if years_vals:
        stats_ax.plot(years_vals,speed_vals,label='Speed')
        stats_ax.plot(years_vals,vision_vals,label='Vision')
        stats_ax.plot(years_vals,size_vals,label='Size')
        stats_ax.plot(years_vals,pop_vals,label='Population')

        stats_ax.set_xlim(0,max(10,total_years))
        all_vals=speed_vals+vision_vals+size_vals+pop_vals
        if all_vals:
            min_y=max(0,min(all_vals)-2)
            max_y=max(all_vals)+4
            stats_ax.set_ylim(min_y,max_y)
        else:
            stats_ax.set_ylim(0,10)
    stats_ax.set_title("Traits & Population Over Evolutionary Time")
    stats_ax.legend(loc='upper left',fontsize='small')

    ax.set_title(f"Time: {format_years(total_years)} | Creatures: {len(creatures)}, Food: {len(food)} {'[PAUSED]' if paused else ''}")

    return [scat,food_scat]+heading_patches[:len(creatures)]+trail_lines[:len(creatures)]

ani = animation.FuncAnimation(fig,update,interval=50,blit=False,cache_frame_data=False)
plt.tight_layout()
plt.show()