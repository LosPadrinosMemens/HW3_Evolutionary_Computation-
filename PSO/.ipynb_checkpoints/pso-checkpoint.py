import numpy as np
from PSO import *
import configparser
from Problems import *


class PSO:
    def __init__(self, objective_function, n_var=2, config_file = "inputs/param_swarm.cfg"):
        self.load_config(config_file)
        self.objective_function = problems.FunctionFactory.select_function(objective_function, n_var)
        self.particle_factory = ParticleFactory(self.objective_function)       
        self.swarm = Swarm(self.swarm_size, self.particle_factory)
        self.lbest = Swarm(self.swarm_size, self.particle_factory)
        self.gbest = self.particle_factory.create_particle()
        # Initialization
        self.gbest.initialize_location(np.inf)
        self.swarm.initialize_swarm()
        self.lbest.initialize_lbest_swarm()
        self.generation_statistics = {}
        self.generation_t = 0
    
    def load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.swarm_size = int(config['SwarmSettings']['swarm_size'])
        self.c1 = float(config['SwarmSettings']['c1'])
        self.c2 = float(config['SwarmSettings']['c2']) 
        self.w = float(config['SwarmSettings']['inertia_factor']) 
        self.Vmax = float(config['SwarmSettings']['Vmax'])
        self.max_generations = int(config['SwarmSettings']['max_generations'])

    def pass_next_generation(self):
        best_fitness =  self.gbest.get_objective_value()
        
        self.generation_t += 1

        self.generation_statistics[self.generation_t] = {
            "best_fitness": best_fitness
        }

    def get_population_statistics(self):
        """
        Returns dictionary with information about best individual across generations
        """
        return self.generation_statistics

    def run(self):
        while self.generation_t  < self.max_generations:
            for i in range(self.swarm.get_swarm_size()):
                particle = self.swarm.get_particle_at(i)
                particle_lbest = self.lbest.get_particle_at(i)
                # Set the personal best position
                if particle.get_objective_value() < particle_lbest.get_objective_value():
                    particle_lbest.set_x(particle.get_x())
                    particle_lbest.set_objective_value(particle.get_objective_value())
                    
                # Update the gBest position
                if particle_lbest.get_objective_value() < self.gbest.get_objective_value():
                    self.gbest.set_x(particle_lbest.get_x())
                    self.gbest.set_objective_value(particle_lbest.get_objective_value())
                    
            # Random numbersfor the calculation of the velocity
            r1 = np.random.rand(self.objective_function.get_nvar(), 1)
            r2 = np.random.rand(self.objective_function.get_nvar(), 1)

            # For each particle, update its velocity and position
            for i in range(self.swarm.get_swarm_size()):
                particle = self.swarm.get_particle_at(i) # x_i
                lbest = self.lbest.get_particle_at(i) # y_i
                # gbest = y^_i
                for j in range(self.objective_function.get_nvar()):
                    particle_j = particle.get_x_at(j)
                    lbest_j =  lbest.get_x_at(j)
                    gbest_j = self.gbest.get_x_at(j)
                    cognitive_comp = self.c1 * r1[j] * (lbest_j - particle_j)
                    social_comp = self.c2 * r2[j] * (gbest_j - particle_j)
                    veloc_0 = particle.get_velocity_at(j)
                    veloc = self.w * veloc_0 + cognitive_comp + social_comp # Inertia weight
                    veloc = min(veloc, self.Vmax) # Velocity clamping
                    value = particle_j + veloc
                    particle.set_velocity_at(veloc,j) # Saving new velocity
                    particle.set_x_at(value, j) # value is the new position for the j-th component of the particle
                particle.evaluate_objective_function() # Calculate the objective value based on the new position of the particle
            self.pass_next_generation()