print("Importing libraries...")
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from helper_functions import *


print("Opening Root File...")
file = uproot.open("events.root")
tree = file["Delphes;1"]
particles = tree['Particle']

print("Unpacking all PIDs...")
particles_PID_by_event = particles['Particle.PID'].array(library='pd')

counts = dict()
for event_num in tqdm(range(50000), desc="Counting", unit="Event #"):
    event_particles = particles_PID_by_event[event_num]
    for PID in event_particles:
        particle = convert_PID_to_string(PID) 
        counts[particle] = counts.get(particle, 0) + 1

print("Trimming Counts...")
new_counts = dict()
threshhold = 100
total_w_threshold = 0
total_wo_threshhold = 0

for count, PID in zip(counts.values(), counts.keys()):
    total_wo_threshhold += count
    if count > threshhold:
        total_w_threshold += threshhold
        new_counts[PID] = count

# Data preparation
trimmed_particles = list(new_counts.keys())
trimmed_counts = list(new_counts.values())

print("Generating Histogram...")
# Create a histogram
plt.figure(figsize=(30, 15))
plt.bar(trimmed_particles, trimmed_counts, color='skyblue')
plt.xlabel('Particle Type')
plt.ylabel('Count')
plt.title('Particle Counts Histogram')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()

# Show the plot
print("Saved Histogram as 'particle_counts.png'")
plt.savefig("particle_counts")