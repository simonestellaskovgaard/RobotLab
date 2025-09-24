import numpy as np
import matplotlib.pyplot as plt

class LandmarkOccupancyGrid:
    def __init__(self, low=(0, 0), high=(2, 2), res=0.05) -> None:
        self.map_area = [low, high]    #a rectangular area    
        self.map_size = np.array([high[0]-low[0], high[1]-low[1]])
        self.resolution = res

        self.n_grids = [ int(s//res) for s in self.map_size]

        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8)

        self.extent = [self.map_area[0][0], self.map_area[1][0], self.map_area[0][1], self.map_area[1][1]]

    def in_collision(self, indices):
        """
        find if the position is occupied or not. return if the queried pos is outside the map
        """
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= self.n_grids[i]:
                return 1
        
        return self.grid[indices[0], indices[1]] 
    
    def robot_collision(self, pos, r_robot):
        """
        Checks if a robot with radius r_robot at position (x, y) collides with an obstacle
        """
        indices = [int((pos[i] - self.map_area[0][i]) // self.resolution) for i in range(2)]

        cell_radius = int(np.ceil(r_robot / self.resolution))

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                neighbor_indices = [indices[0] + dx, indices[1] + dy]
                
                if self.in_collision(neighbor_indices):
                    return 1  
        
        return 0  


    def add_landmarks(self, landmarks):
        """
        Fill the grid with landmarks.
        Each landmark is given as (x, y, r) where
        - (x, y) is the center in world coordinates
        - r is the landmark's radius
        """
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array([
                    self.map_area[0][0] + self.resolution * (i + 0.5),
                    self.map_area[0][1] + self.resolution * (j + 0.5)
                ])
                
                for (x, y, r) in landmarks:
                    if np.linalg.norm(centroid - np.array([x, y])) <= r:
                        self.grid[i, j] = 1
                        break  


    
    def draw_map(self, robot_radius=0.25):
        plt.imshow(self.grid.T, cmap="Greys", origin='lower', vmin=0, vmax=1, extent=self.extent, interpolation='none')
        if robot_radius is not None:
            circle = plt.Circle((0,0), robot_radius, color='blue', alpha=0.5, label='Robot')
            plt.gca().add_patch(circle)
        plt.legend()

if __name__ == '__main__':
    grid_map = LandmarkOccupancyGrid(low=(-2,-2), high=(2,2), res=0.05)
    landmarks = [
        (0.5, 0.5, 0.2),
        (1.2, 1.0, 0.15),
        (1.8, 1.5, 0.25)
    ]
    grid_map.add_landmarks(landmarks)

    robot_radius = 0.1

    plt.figure(figsize=(5,5))
    grid_map.draw_map( robot_radius=robot_radius)
    plt.show()

