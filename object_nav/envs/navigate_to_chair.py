from mini_behavior.roomgrid import *
from mini_behavior.register import register
from mini_behavior.objects import *
import copy
import random


class NavigateToObj(RoomGrid):
    """
    Environment simulate an agent navigate to a specific obj
    """

    def __init__(self,
                 mode='primitive',
                 initial_dict=None,
                 max_steps=1e5):
        # todo: make the mission random from this list
        self.mission_list = ["chair", "bed", "refrigerator"]

        self.mission = initial_dict["Grid"]["mission"]

        # configurations for agent generation
        self.initial_dict = initial_dict
        self.state_dict = copy.deepcopy(initial_dict)

        # for auto nums
        self.auto_room_split_dirs = self.initial_dict["Grid"]["auto"]["room_split_dirs"]
        self.auto_min_room_dim = self.initial_dict["Grid"]["auto"]["min_room_dim"]
        self.auto_max_num_room = self.initial_dict["Grid"]["auto"]["max_num_room"]
        # agents
        self.agents = []

        # generate floorplans(i.e rooms), objects, agents
        super().__init__(mode=mode,
                         init_dict=initial_dict,
                         max_steps=max_steps,
                         )

    def _reward(self):
        return 0

    def _init_conditions(self):
        pass

    def _end_conditions(self):
        chairs: list[WorldObj] = self.objs['chair']
        for chair in chairs:
            if chair.check_abs_state(self, 'inreachofrobot'):
                return True
        return False


register(
    id='MiniGrid-NavChair-16x16-N2-v0',
    entry_point='object_nav.envs:NavigateToObj'
)

