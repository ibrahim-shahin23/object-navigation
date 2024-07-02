"""
Create abstract class Rewarder. This class will be used to implement different reward functions
using Strategy Pattern
"""
import object_nav.envs
# from object_nav.envs.navigate_to_chair import NavigateToObj
from mini_behavior.objects import WorldObj

import numpy as np
import matplotlib.pyplot as plt


class rewarder:
    def __init__(self):
        pass

    def get_reward(self, env):  # env: NavigateToObj
        """should be implemented by the subclass"""
        raise NotImplementedError

    def reset(self, env):  # env: NavigateToObj
        """should be implemented by the subclass"""
        raise NotImplementedError


class distance_rw(rewarder):
    def __init__(self):
        super().__init__()
        self.goals = []
        self.goals_pos = []
        self.agent_pos = None

        self.grid: np.array = None

    def get_reward(self, env):  # env: NavigateToObj
        new_agent_pos = env.agent_pos
        value = self.grid[new_agent_pos[1], new_agent_pos[0]]\
            - self.grid[self.agent_pos[1], self.agent_pos[0]]
        self.agent_pos = new_agent_pos
        if value > 0:
            value *=0.9
        elif value < 0:
            value *=0.5
        return value

    def reset(self, env):  # env: NavigateToObj
        """
        Reset the rewarder.
        it should be called at the beginning of each episode.
        it rebuilds a copy of the env grid and assign values to each cell based on
        the distance to the goal using following metric:
        1. cells with the goal object have the largest values
        2. 4-connected cells have the same value
        3. cells with obstacles have the negative value
        :param env:
        :return:
        """
        self.goals_pos = []
        self.goals: [WorldObj] = env.objs[env.goal_obj]
        self.agent_pos = env.agent_pos

        self._build_grid(env)

        # show the grid as heat map using matplotlib
        # plt.imshow(self.grid, cmap='hot', interpolation='nearest')
        # plt.show()
        # print(self.grid)

    def _build_grid(self, env):
        """
        Build a copy of the env grid and assign values to each cell based on the distance to the goal
        :param env:
        :return:
        """
        self.grid = np.zeros((env.height, env.width))

        self.values = {'obstacle': -20, 'goal': 20}

        # 1) assign -20 for all cells containing obstacles
        for i in range(env.width):
            for j in range(env.height):
                if not env.grid.is_empty(i, j):
                    self.grid[j, i] = self.values['obstacle']

        # 2) loop over goals and assign the largest value to the cells containing the goal object
        for goal in self.goals:
            # if it is furniture, we need to assign the value to the cells containing the furniture
            if goal.is_furniture():
                for pos in goal.all_pos:
                    self.grid[pos[::-1]] = self.values['goal']
                    self.goals_pos.append(pos[::-1])
            else:
                self.grid[goal.init_pos[::-1]] = self.values['goal']
                self.goals_pos.append(goal.init_pos[::-1])

        # 3) assign decreasing values to cells based on the steps away from the goals
        for goal_pos in self.goals_pos:
            # print(goal_pos)
            self._bfs_custom(goal_pos[0], goal_pos[1], 20)

    def _bfs_custom(self, x, y, value):
        step = 1
        queue = [{"pt": (x, y), "step": step}]

        visited_cells = set()
        while queue:
            node = queue.pop(0)
            x, y = node['pt']
            step = node["step"]
            visited_cells.add((x, y))

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) in visited_cells:
                    continue

                if (0 <= new_x < self.grid.shape[1] and 0 <= new_y < self.grid.shape[0]) \
                        and (self.grid[new_x, new_y] not in self.values.values()):
                    # check if the value is larger than the current value
                    if self.grid[new_x, new_y] < value - step:
                        self.grid[new_x, new_y] = value - step if value - step > 0 else 0
                    queue.append({"pt": (new_x, new_y), "step": step + 1})


class steps_rw(rewarder):
    def __init__(self):
        super().__init__()

    def get_reward(self, env):  # env: NavigateToObj
        if not env.action_done:
            return -0.5
        else:
            return -0.01

    def reset(self, env):  # env: NavigateToObj
        pass


class composite_rw(rewarder):
    def __init__(self, rw_list: [rewarder]=[]):
        super().__init__()
        self.rw_list = rw_list

    def get_reward(self, env):  # env: NavigateToObj
        return sum([rw.get_reward(env) for rw in self.rw_list])

    def reset(self, env):  # env: NavigateToObj
        for rw in self.rw_list:
            rw.reset(env)

    def add_rw(self, rw: rewarder):
        self.rw_list.append(rw)


if __name__ == '__main__':
    pass
