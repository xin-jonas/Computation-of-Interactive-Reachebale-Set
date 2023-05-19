"""IDM behavior model for CommonRoad agents.
Inspired from: Matthias Rowold
Modified by: Aroua Bel Haj Amor
"""


from commonroad.scenario.scenario import Scenario
from commonroad_helper_functions.spacial import lanelet2spline
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D
# from motion_planner_components.prediction.advanced_models.agent_sim.agent import Agent
from agent import Agent
import numpy as np
import os
import sys

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(module_path)

class IDMAgent(Agent):
    """IDM Agent.

    Class to model IDM behavior
    """

    def __init__(
        self,
        scenario: Scenario,
        agent_id: int,
        expected_lanelets_list: list = None,
        ref_CLCS = None,
        enable_logging: bool = True,
        log_path: str = '/log',
        debug_step: bool = False,
        idm_parameters: dict = {
            'v_0': 20,
            's_0': 4,
            'T': 2,
            'a_max': 3,
            'a_min': -4,
            'b': 1.5,
            'delta': 4,
            'label': 2,
            'action': None
        },
    ):
        """Initialize an IDM agent.

        :param scenario: commonroad scenario
        :param agent_id: ID of the IDM agent: should be equal to the commonroad dynamic obstacle ID
        :param enable_logging: True for logging
        :param log_path: path for logging files
        :param debug_step: True for figure with current scenario in every time step
        :param idm_parameters: IDM parameters

        """

        # initialize the parent class
        super().__init__(
            scenario=scenario,
            agent_id=agent_id,
            expected_lanelets_list=expected_lanelets_list,
            ref_CLCS=None,
            enable_logging=enable_logging,
            log_path=log_path,
            debug_step=debug_step,
        )

        # idm parameters
        self.__idm_parameters = idm_parameters

    def _step_agent(self, delta_time: float, action_number=None):
        """IDM step.

        This methods overloads the basic step method. It calculates the new state according to the IDM behavior.
        An acceleration depending on the leading vehicle is integrated twice to obtain the new position.

        :param delta_time: time difference to the previous step
        """

        if not self._valid:
            return

        # new acceleration
        if self.idm_parameters['label'] == 0:
            # ego
            acceleration = self.take_actions(self.idm_parameters['action'])
        elif self.idm_parameters['label'] == 1:
            # direct effect the ego
            acceleration = self.__get_acceleration()
        elif self.idm_parameters['label'] == 2:
            # effected with ego
            acceleration = self.__get_acceleration()
        elif self.idm_parameters['label'] == 3:
            # undierected effected with ego
            acc_des = (self.idm_parameters['v_0'] - self.state.velocity)/delta_time
            if acc_des > self.idm_parameters['a_max']:
                acc_des = self.idm_parameters['a_max']
            elif acc_des < self.idm_parameters['a_min']:
                acc_des = self.idm_parameters['a_min']
            acceleration = acc_des
        else:
            acceleration = self.__get_acceleration()

        # orientation
        alpha = self.state.orientation
        # new velocity
        velocity = self.state.velocity + acceleration * delta_time

        # covered distance along the center line of the current lanelet
        ds = self.state.velocity * delta_time + 1 / 2 * acceleration * delta_time ** 2

        # approximate the center line of the current lanelet as a cubic spline
        # print(self.current_lanelet_id)
        center_points = np.array(self.CLCS_main.reference_path())
        try:
            ego_lanelet_spline = CubicSpline2D(center_points[:, 0], center_points[:, 1])
        except ValueError:
            center_points = np.unique(center_points, axis=0)
            ego_lanelet_spline = CubicSpline2D(center_points[:, 0], center_points[:, 1])

        # lanelettemp=self.scenario.lanelet_network.find_lanelet_by_id(
        #         self.current_lanelet_id
        #     )
        # ego_lanelet_spline = lanelet2spline(
        #     lanelet=lanelettemp
        # )

        # calculate the new position (arc length) travelled along the spline
        s_new = ego_lanelet_spline.get_min_arc_length(self.state.position)[0] + ds

        # new position
        x, y = ego_lanelet_spline.calc_position(s_new)
        position = np.array([x, y])

        # new orientation
        orientation = ego_lanelet_spline.calc_yaw(s_new)
        # if orientation > 1.58:
        #     print('A')
        # update the state
        self._state.position = position
        self._state.orientation = orientation
        self._state.velocity = velocity
        self._state.acceleration = acceleration

    def __get_acceleration(self):
        """Get acceleration.

        This method calculates the new acceleration depending on the leading vehicle and the desired velocity

        :return: acceleration in m/s^2
        """
        # standstill
        if self.idm_parameters['v_0'] == 0:
            if self._state.velocity > 0:
                return self.idm_parameters['a_min']
            else:
                return 0

        # free road term
        a_free = self.idm_parameters['a_max'] * (
                1
                - (self.state.velocity / self.idm_parameters['v_0'])
                ** self.idm_parameters['delta']
        )

        # interaction term
        if self.leader_id is not None:
            a_int = (
                -self.idm_parameters['a_max']
                * (
                    (
                        self.idm_parameters['s_0']
                        + self.state.velocity * self.idm_parameters['T']
                    )
                    / self.distance_to_leader
                    + self.state.velocity
                    * self.approaching_rate_to_leader
                    / (
                        2
                        * np.sqrt(
                            self.idm_parameters['a_max'] * self.idm_parameters['b']
                        )
                        * self.distance_to_leader
                    )
                )
                ** 2
            )
        else:
            a_int = 0

        # # disable going backwards
        # if self.state.velocity <= 0 and (a_free + a_int) <= 0:
        #     return 0

        # disable going backwards
        if self.state.velocity + (a_free + a_int)*self.scenario.dt <= 0:
            return - self.state.velocity/self.scenario.dt

        # force convergence to desired speed
        if self.state.velocity + (a_free + a_int)*self.scenario.dt >= self.idm_parameters['v_0']:
            return (self.idm_parameters['v_0'] - self.state.velocity)/self.scenario.dt

        return max(a_free + a_int, self.idm_parameters['a_min'])

    def take_actions(self, action_number):
        sample = False
        if sample:
            if action_number == 0:
                acc = np.random.choice(np.linspace(-1, 1, num=5), 1)[0]
            elif action_number == 1:
                acc = np.random.choice(np.linspace(1, 4, num=5), 1)[0]
            elif action_number == 2:
                acc = np.random.choice(np.linspace(4, 7, num=5), 1)[0]
            elif action_number == -1:
                acc = np.random.choice(np.linspace(-3, -1, num=5), 1)[0]
            elif action_number == -2:
                acc = np.random.choice(np.linspace(-6, -3, num=5), 1)[0]
        else:
            if action_number == 0:
                acc = 0
            elif action_number == 1:
                acc = self.idm_parameters['a_max']
            elif action_number == 2:
                acc = self.idm_parameters['a_min']/2
            else:
                acc = self.idm_parameters['a_min']

        return acc
    ####################################################################################################################
    # PROPERTIES #######################################################################################################
    ####################################################################################################################

    @property
    def idm_parameters(self):
        """IDM parameters."""
        return self.__idm_parameters


# EOF
