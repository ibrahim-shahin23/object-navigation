from mini_behavior.roomgrid import *
from mini_behavior.register import register


class TwoRoomNavigationEnv(RoomGrid):
    """
    Environment in which the agent is instructed to navigate to a target position
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        pickup_0 = 9
        pickup_1 = 10
        pickup_2 = 11
        drop_0 = 12
        drop_1 = 13
        drop_2 = 14

    def __init__(
            self,
            max_steps=1e5,
    ):
        self.mode = 'primitive'
        super().__init__(mode=self.mode,
                         num_objs={'ball': 1},
                         room_size=8,
                         num_rows=1,
                         num_cols=2,
                         max_steps=max_steps,
                         see_through_walls=False,
                         agent_view_size=3,
                         highlight=False
                         )

        self.actions = TwoRoomNavigationEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        # print(list(self.actions), '\n', len(self.actions))

    def _gen_grid(self, width, height):
        self._gen_rooms(width, height)
        # randomize the agent start position and orientation
        self._gen_objs()
        self.place_agent()
        self.connect_all()
        self.mission = 'navigate between rooms'

    def _gen_objs(self):
        for obj in self.obj_instances.values():
            self.place_obj(obj)

    def _end_conditions(self):
        ball: WorldObj = self.objs['ball'][0]
        if ball.check_abs_state(env=self, state='inhandofrobot'):
            return True
        return False


register(
    id='MiniGrid-TwoRoomNavigation-8x8-N2-v0',
    entry_point='mini_behavior.envs:TwoRoomNavigationEnv'
)
