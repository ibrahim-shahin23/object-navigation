from mini_behavior.roomgrid import *
from mini_behavior.register import register


class NavigationEnv(RoomGrid):
    """
    Environment in which the agent is instructed to navigate to a target position
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            num_objs=None
    ):
        if num_objs is None:
            num_objs = {'goal': 1}

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_grid(self, width, height):
        self._gen_rooms(width, height)

        # generate goal position
        goal = self.objs['goal'][0]
        _, self.target_pos = self.place_in_room(0, 0, goal, reject_fn=None)
        goal.on_floor = True

        # generate other objects
        if 'ball' in self.objs.keys():
            balls = self.objs['ball']
            for ball in balls:
                self.place_in_room(0, 0, ball, reject_fn=None)

        # randomize the agent start position and orientation
        self.place_agent()
        self.mission = 'navigate to the target position'
        self.connect_all()

    def _end_conditions(self):
        if np.all(self.agent.cur_pos == self.target_pos):
            return True
        else:
            return False


class NavigationEnv16x16_Human(NavigationEnv):
    def __init__(self):
        super().__init__(mode='cartesian',
                         room_size=16,
                         )


class NavigationMultiEnv16x16_Human(NavigationEnv):
    def __init__(self):
        super().__init__(mode='cartesian',
                         room_size=16,
                         num_rows=2,
                         num_cols=2
                         )


class NavigationMultiEnv16x16_RL(NavigationEnv):
    def __init__(self):
        super().__init__(mode='primitive',
                         room_size=16,
                         num_rows=2,
                         num_cols=2
                         )

class NavigationMultiEnv8x8_Human(NavigationEnv):
    def __init__(self):
        super().__init__(mode='cartesian',
                         room_size=8,
                         num_rows=2,
                         num_cols=2
                         )


# human control env
register(
    id='MiniGrid-Navigation-16x16-N1-v0',
    entry_point='mini_behavior.envs:NavigationEnv16x16_Human'
)


##### MULTI ROOM
# human control env
register(
    id='MiniGrid-NavigationMulti-16x16-N1-v0',
    entry_point='mini_behavior.envs:NavigatioMultiEnv16x16_Human'
)

# RL agent env
register(
    id='MiniGrid-NavigationMulti-16x16-N2-v0',
    entry_point='mini_behavior.envs:NavigationMultiEnv16x16_RL'
)

# human control env
register(
    id='MiniGrid-NavigationMulti-8x8-N1-v0',
    entry_point='mini_behavior.envs:NavigationMultiEnv8x8_Human'
)
