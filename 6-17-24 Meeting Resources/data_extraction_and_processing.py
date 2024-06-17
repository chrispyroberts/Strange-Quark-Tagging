print("Importing libraries...")
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from helper_functions import *

print("Opening File...")
# Open the ROOT file
file = uproot.open("events.root")

# Access a tree within the ROOT file
tree = file["Delphes;1"]

particles = tree['Particle']
jets = tree['Jet']

print("Loading file data...")
jet_particles = jets['Jet.Particles'].array() 
jet_flavors =  jets['Jet.Flavor'].array()

# this is a hash map where the key is event index (int) and value is
# a bool indicating a given event has jets
event_has_jets = [True if x > 0 else False for x in jets.array()] 

# need to store 2d arrays as variables, because every time we re-access them it takes a lot of time
# better to just store them in memory and access them from there using indexing
X_array =           particles['Particle.X'].array(library='np')
Y_array =           particles['Particle.Y'].array(library='np')
Z_array =           particles['Particle.Z'].array(library='np')
mass_array =        particles['Particle.Mass'].array(library='np')
energy_array =      particles['Particle.E'].array(library='np')
eta_array =         particles['Particle.Eta'].array(library='np')
phi_array =         particles['Particle.Phi'].array(library='np')
charge_array =      particles['Particle.Charge'].array(library='np')
lifetime_array =    particles['Particle.T'].array(library='np')

jet_pt_array =      jets['Jet.PT'].array(library='np')
jet_eta_array =     jets['Jet.Eta'].array(library='np')
jet_phi_array =     jets['Jet.Phi'].array(library='np')

# need to define this here because the arrays are stored as global variables in the file
# maybe a better way to do this but idrc, it works
def get_particle_variables(event_num, list_of_PID_lists):
    particle_variables = []

    for list_of_particles in list_of_PID_lists:
        tmp = []
        for particle_num in list_of_particles:
            index = particle_num - 1 # 1-indexed

            X = X_array[event_num][index]
            Y = Y_array[event_num][index]
            Z = Z_array[event_num][index]
            
            mass = mass_array[event_num][index]
            energy = energy_array[event_num][index]
            eta = eta_array[event_num][index]
            phi = phi_array[event_num][index]

            charge = charge_array[event_num][index]
            lifetime = lifetime_array[event_num][index]
        
            r_0, eta_0, phi_0 = cartesian_to_radial(X, Y, Z)

            if eta_0 == np.inf:
                eta_0 = np.nan

            tmp.append(tuple([energy, eta, phi, mass, r_0, eta_0, phi_0, charge, lifetime]))

        particle_variables.append(tmp)

    return particle_variables

raw_X_particles = []
raw_X_jets = []
raw_y = []

for event in tqdm(range(50000), desc="Collecting raw data...", unit="Event #"):
    # check if there are jets in this event
    if event_has_jets[event]:
        # if yes, for each jet find the particles in it and the flavor of the jet
        
        id_lists = []
        flavor_list = []
        jet_data_lists = []
        
        # get all data from jets within the same event so we don't have to access
        # the tree structure many times
        for flavor, ref_array, jet_pt, jet_eta, jet_phi in zip(jet_flavors[event], 
                                                                jet_particles[event], 
                                                                jet_pt_array[event],
                                                                jet_eta_array[event],
                                                                jet_phi_array[event]):
            jet_data = tuple([jet_pt, jet_eta, jet_phi])
            jet_data_lists.append(jet_data)
            list_of_particle_IDS = ref_array['refs']
            id_lists.append(list_of_particle_IDS)
            flavor_list.append(flavor)
    
        particle_data = get_particle_variables(event, id_lists)

        for flavor, particle_variables, jet_variables in zip(flavor_list, particle_data, jet_data_lists):
            raw_X_particles.append(particle_variables)
            raw_X_jets.append(jet_variables)
            raw_y.append(flavor)

print(f"{len(raw_X_particles)} particles found")

for particle_data in tqdm(raw_X_particles, desc="Processing particle raw data...", unit="Particle #"):
    while len(particle_data) < 50:
        # keep appending tuple of nan values until the length of each input array is 50x9
        particle_data.append(tuple([np.nan for _ in range(0, 9)]))


# convert flavors to bools
strange_jet_flavor = 3

print("Adding binary labels to jets...")
# 1 for strange, 0 for non-strange
for index, flavor in enumerate(raw_y):
    raw_y[index] = 1 if flavor == strange_jet_flavor else 0

print(f"Saving Processed Data...")
processed_X_particles = np.array(raw_X_particles)
processed_X_jets = np.array(raw_X_jets)
processed_y = np.array(raw_y)

np.save('data/particle_data', processed_X_particles)
print("Particle data saved in 'data/particle_data.npy")

np.save('data/jet_data', processed_X_jets)
print("Jet data saved in 'data/jet_data.npy")

np.save('data/flavor_tags', processed_y)
print("Flavor tags saved in 'data/flavor_tags.npy")

# loading files:
loaded = np.load('data/flavor_tags.npy')