"""Agent class for Commonroad multi-agent simulation.
Inspired from: Matthias Rowold
Modified by: Aroua Bel Haj Amor

"""
from typing import List, Tuple, Union
from commonroad.prediction.prediction import Occupancy
from commonroad.scenario.trajectory import State
from commonroad.visualization.mp_renderer import MPRenderer
from copy import deepcopy
from commonroad.scenario.scenario import Scenario, Lanelet
import numpy as np
from matplotlib import pyplot as plt
from commonroad_helper_functions.logger import ObjectStateLogger
from commonroad_helper_functions.visualization import get_plot_limits_from_scenario
from commonroad_helper_functions.spacial import (
    get_follower_on_lanelet,
    lanelet2spline,
)
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


def get_leader_on_lanelet(
        scenario, ego_obstacle_id, leader_lanelet_id, time_step, lanelet_merged_list, CLCS_main
):
    """Get Leader on Lanelet

        Find the ID of the next leading obstacle on a specified lanelet

        :param scenario: commonroad scenario
        :param ego_obstacle_id: obstacle id of the considered obstacle (ego vehicle) of which to find the leader
        :param leader_lanelet_id: lanelet ID of the lanelet to search for a leader
        :param time_step: scenario time step
        :return: ID of the leading obstacle, distance to leader, approaching rate to leader
        """
    # print(CLCS)
    # print(lanelet_merged_list[0])
    # leader_lanelet = scenario.lanelet_network.find_lanelet_by_id(leader_lanelet_id)
    leader_id = None
    approaching_rate = None
    distance = 1000
    #  get position of vehicle (no ego)
    ego_vehicle = scenario.obstacle_by_id(ego_obstacle_id)
    # ego_position = ego_vehicle.prediction.trajectory.state_list[time_step].position
    dynamic_obstacle_lst = [ob for ob in scenario.dynamic_obstacles if (ob.obstacle_id != ego_obstacle_id)]
    if time_step > 0:
        dynamic_obstacle_position = [ob.prediction.trajectory.state_list[time_step-1].position for ob in
                                     dynamic_obstacle_lst]
    else:
        dynamic_obstacle_position = [ob.initial_state.position for ob in
                                     dynamic_obstacle_lst]

    if dynamic_obstacle_position:
        dynamic_obstacle_position.append([10000, 10000])
        dynamic_obstacle_position = np.array(dynamic_obstacle_position)
    else:
        return leader_id, None, approaching_rate

    # lane_ref = [scenario.lanelet_network.find_lanelet_by_id(lanelet_id) for lanelet_id in lanelet_merged_list]
    for lanelet_element in lanelet_merged_list:

        in_lanelet_lst = lanelet_element.contains_points(dynamic_obstacle_position)
        obs_index_in_lanelet = np.where(np.array(in_lanelet_lst) == True)[0]
        obs_in_lanelet = [dynamic_obstacle_lst[ob_index] for ob_index in obs_index_in_lanelet]

        if obs_in_lanelet:
            # if ego_obstacle_id == 328:
            #     for ob in obs_in_lanelet:
            #         print('obstacle in lanelets with id ' + repr(lanelet_element.lanelet_id) + ':' + repr(ob.obstacle_id))

            # approximate center lane with a cubic spline
            lanelet_spline = lanelet2spline(lanelet=lanelet_element)
            ego_arclength = lanelet_spline.get_min_arc_length(
                ego_vehicle.prediction.trajectory.state_list[time_step - 1].position
            )[0]

            # turn the cartesian coordinate to cur of ego state
            # state_ego = ego_vehicle.prediction.trajectory.state_list[time_step]
            # dist = []
            # CLCS = CLCS_main
            # try:
            #     p_lon_ego, _ = CLCS.convert_to_curvilinear_coords(state_ego.position[0], state_ego.position[1])
            # except ValueError:
            #     print('aa')
            # for d_ob in obs_in_lanelet:
            #     if time_step > 0:
            #         state_ob = d_ob.prediction.trajectory.state_list[time_step]
            #     else:
            #         state_ob = d_ob.initial_state
            #     try:
            #         p_lon_ob, _ = CLCS.convert_to_curvilinear_coords(state_ob.position[0], state_ob.position[1])
            #         dist_to_obs = (p_lon_ob - d_ob.obstacle_shape.length / 2) - \
            #                       (p_lon_ego + ego_vehicle.obstacle_shape.length / 2)
            #         dist.append(dist_to_obs)
            #     except ValueError:
            #         pass


            if time_step > 0:
                # TODO: include lengths of the vehicles
                dist = [
                    lanelet_spline.get_min_arc_length(
                        d_ob.prediction.trajectory.state_list[time_step - 1].position
                    )[0] # - d_ob.obstacle_shape.length / 2
                    - ego_arclength #- ego_vehicle.obstacle_shape.length / 2
                    for d_ob in obs_in_lanelet
                ]
            else:
                # TODO: include lengths of the vehicles
                dist = [
                    lanelet_spline.get_min_arc_length(
                        d_ob.initial_state.position
                    )[0] - d_ob.obstacle_shape.length / 2
                    - ego_arclength - ego_vehicle.obstacle_shape.length/2
                    for d_ob in obs_in_lanelet
                ]
            if ego_obstacle_id == 328:
                print(dist)

            if ego_obstacle_id == 31:
                s = 'the distance to leader of ego vehicle with id: 31 ' + repr(dist)
                print(s)

            dist_positive = [d for d in dist if (0 < d < distance)]

            if dist_positive:
                distance = min(dist_positive)
                leader_index = dist.index(distance)
                leader_id = obs_in_lanelet[leader_index].obstacle_id
                # if ego_obstacle_id  == 328:
                s = 'The ego vehicle is ' + repr(ego_obstacle_id) + ', leader vehicle is ' + repr(leader_id) \
                + ', and distance is ' + repr(distance)
                # print(s)
            # distance = distance - ego_vehicle.obstacle_shape.length
            if leader_id is not None:
                # calculate the approaching rate
                if time_step > 0:
                    orien_diff = ego_vehicle.prediction.trajectory.state_list[time_step - 1].orientation - \
                                 scenario.obstacle_by_id(leader_id).prediction.trajectory.state_list[time_step - 1].orientation

                    approaching_rate = (
                            ego_vehicle.prediction.trajectory.state_list[time_step - 1].velocity
                            - scenario.obstacle_by_id(leader_id)
                            .prediction.trajectory.state_list[time_step - 1]
                            .velocity*np.cos(-orien_diff)
                    )

                else:
                    orien_diff = ego_vehicle.initial_state.orientation\
                                 - scenario.obstacle_by_id(leader_id).initial_state.orientation

                    approaching_rate = (
                            ego_vehicle.initial_state.velocity
                            - scenario.obstacle_by_id(leader_id).initial_state.velocity*np.cos(-orien_diff)
                    )
                # if ego_obstacle_id == 328:
                s = 'approaching_rate is ' + repr(approaching_rate) + ', orientation diff is ' + repr(orien_diff) \
                    + ', cos: ' + repr(np.cos(orien_diff))
                # print(s)

    if distance == 1000:
        distance = None

    return leader_id, distance, approaching_rate


class Agent(object):
    """Agent class for CommonRoad.

    Class to represent agents in a scenario.
    """

    def __init__(
            self,
            scenario: Scenario,
            agent_id: int,
            expected_lanelets_list: list,
            ref_CLCS = None,
            enable_logging: bool = False,
            log_path: str = '/log',
            debug_step: bool = False,
    ):
        """Initialize an agent.
        :param scenario: commonroad scenario
        :param agent_id: ID of the agent: should be equal to the commonroad dynamic obstacle ID
        :param enable_logging: True for logging
        :param log_path: path for logging files
        :param debug_step: True for figure with current scenario in every time step
        """

        # commonroad scenario in which the agent is moving
        self.__scenario = scenario

        # agent ID
        self.__agent_id = agent_id

        # agent shape
        self.__agent_shape = self.scenario.obstacle_by_id(self.agent_id).obstacle_shape

        # initial lanelets
        self.obstacle = self.scenario.obstacle_by_id(self.agent_id)
        self.occ_current = self.obstacle.occupancy_at_time(self.obstacle.initial_state.time_step)
        self.set_ids_lanelets_current = \
            self.__scenario.lanelet_network.find_lanelet_by_shape(self.occ_current.shape)

        # expected reference lanelets defined by human
        self.expected_lanelets_list = expected_lanelets_list

        if expected_lanelets_list:
            expected_reference_lanelets = int("".join([str(i) for i in expected_lanelets_list]))
        else:
            expected_reference_lanelets = None

        self.expected_reference_lanelets = expected_reference_lanelets

        # merged lanelets
        self.list_lanelets_merged = list()
        self.list_CLCS: List[CurvilinearCoordinateSystem] = list()
        self._retrieve_merged_lanelet_ids_and_create_CLCS(scenario, agent_id,
                                                          self.set_ids_lanelets_current, self.expected_reference_lanelets)

        if ref_CLCS:
            self.CLCS_main = ref_CLCS
        else:
            self.CLCS_main = self.list_CLCS[0]

        # initial state
        self.__initial_state = self.scenario.obstacle_by_id(self.agent_id).initial_state

        # current state
        self._state = deepcopy(self.__initial_state)

        # predefined state_list
        self._predefined_state_list = self.scenario.obstacle_by_id(
            self.agent_id
        ).prediction.trajectory.state_list

        # simulation time step size
        self.__dt = scenario.dt
        # current simulation time step
        self._time_step = 0
        # current simulation time
        self.__time = 0.0

        # validity
        self._valid = True

        # initial lanelet
        self.__current_lanelet_id = None
        self.update_current_lanelet()

        # initial leader
        self.__leader_id = (
            self.__distance_to_leader
        ) = self.__approaching_rate_to_leader = None
        self.update_leader()

        # initial follower
        self.__follower_id = (
            self.__distance_to_follower
        ) = self.__approaching_rate_of_follower = None
        # self.update_follower()

        # debugging
        self.__debug_step = debug_step

        # debugging figure
        if self.debug_step:
            self.plot_debug()

        # logging
        self.__logging_enabled = enable_logging
        self.__log_path = log_path

        if self.logging_enabled:
            # create logging object
            self._agent_state_logger = ObjectStateLogger(
                log_path=self.log_path, object_id=self.agent_id
            )
            # initialize the logger for writing
            self._agent_state_logger.initialize()

            # log initial state
            self._log_agent_state()

    def _retrieve_merged_lanelet_ids_and_create_CLCS(self, scenario: Scenario, ego_obstacle_id: int,
                                                     set_ids_lanelets_current, expected_reference_lanelets):
        # print('length of current id: ' + repr(len(set_ids_lanelets_current)))
        if self.expected_lanelets_list:
            set_ids_lanelets_current = [i for i in set_ids_lanelets_current if i in self.expected_lanelets_list]
            lanelet_current = scenario.lanelet_network.find_lanelet_by_id(set_ids_lanelets_current[0])
            list_lanelets_merged = \
                Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                    lanelet=lanelet_current,
                    network=scenario.lanelet_network)[0]

            if list_lanelets_merged is None:
                list_lanelets_merged = [lanelet_current]

            for lanelet in list_lanelets_merged:
                if lanelet.lanelet_id == expected_reference_lanelets:
                    self.list_lanelets_merged = [lanelet]

            if len(self.list_lanelets_merged)==0:
                self.list_lanelets_merged = list_lanelets_merged

            for lanelet in self.list_lanelets_merged:
                CLCS = CurvilinearCoordinateSystem(lanelet.center_vertices)
                CLCS.compute_and_set_curvature()
                self.list_CLCS.append(CLCS)

        else:
            # list_lanelets_merged = None
            for current_lanelet_id in set_ids_lanelets_current[0:1]:
                # if current_lanelet_id == self.expected_lanelets_list[0] if self.expected_lanelets_list else True:
                # print(current_lanelet_id)
                lanelet_current = scenario.lanelet_network.find_lanelet_by_id(current_lanelet_id)
                list_lanelets_merged = \
                    Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                        lanelet=lanelet_current,
                        network=scenario.lanelet_network)[0]
                    # print([l.lanelet_id for l in list_lanelets_merged])
                if list_lanelets_merged is None:
                    list_lanelets_merged = [lanelet_current]

                for lanelet in list_lanelets_merged:
                    CLCS = CurvilinearCoordinateSystem(lanelet.center_vertices)
                    CLCS.compute_and_set_curvature()
                    self.list_CLCS.append(CLCS)

    def step(self, scenario: Scenario):
        """Main step function.

        This step method is called for every agent moving in the scenario.
        It is a wrapper for the agent-type depending actual step method "_step_agent()" that updates the state.
        "_step_agent()" must be overloaded by inheriting classes that implement a certain agent behavior or plan trajectories.

        :param scenario: current commonroad scenario
        """

        # get current commonroad scenario
        self.__scenario = scenario

        # update lanelet
        self.update_current_lanelet()

        # if the agent is inside the lanelet network and valid
        if self.current_lanelet_id is not None and self._valid:
            # update leader
            self.update_leader()
        else:
            self._valid = False

        # debugging figure
        if self.debug_step:
            self.plot_debug()

        # save the current time step temporarily
        time_step_temp = self.time_step

        # update the state depending on the behavior or planned trajectory
        self._step_agent(delta_time=self.scenario.dt)

        # increase the time step by one (this ensures that the time step is not changed by _step_agent())
        self._time_step = time_step_temp + 1

        # ensure correct time step of the new state
        self._state.time_step = self.time_step

        # simulation time
        self.__time = self.time_step * self.dt

        # log the current state
        self._log_agent_state()

    def step_agent_without_idm(self, sim_scenario, state):
        # get current commonroad scenario
        self.__scenario = sim_scenario
        # self.update_current_lanelet()
        # if the agent is inside the lanelet network and valid
        # if self.current_lanelet_id is not None and self._valid:
        #     # update leader
        #     self.update_leader()
        # else:
        #     self._valid = False
        # save the current time step temporarily
        time_step_temp = self.time_step
        # update the state depending on the behavior or planned trajectory
        self._step_agent_without_idm(state)
        # increase the time step by one (this ensures that the time step is not changed by _step_agent())
        self._time_step = time_step_temp + 1
        # ensure correct time step of the new state
        self._state.time_step = self.time_step
        # simulation time
        self.__time = self.time_step * self.dt

    def _step_agent_without_idm(self, state):
        # agent_ob_traj = scenario.obstacle_by_id(self.agent_id).prediction.trajectory
        # try:
        #     up_state = agent_ob_traj.state_list[time_step]
        # except IndexError:
        #     up_state = agent_ob_traj.state_list[-1]
        self._state.position = state.position
        self._state.orientation = state.orientation
        self._state.velocity = state.velocity
        self._state.acceleration = state.acceleration

    def set_to_time_step(self, time_step: int):
        """Set to time step.

        This function sets an agent to the specified time step.
        """
        self._time_step = time_step
        self.update_current_lanelet()
        if self.current_lanelet_id is not None:
            # update leader
            self.update_leader()
            # update follower
            # self.update_follower()

    def _log_agent_state(self):
        """Log agent state.

        Write the current state to the logging file
        """
        if self.logging_enabled:
            self._agent_state_logger.log_state(state=self.state, time=self.time)

    def _step_agent(self, delta_time):
        """Agent step function.

        This method directly changes the state of the agent.
        It must be overloaded to enforce a desired behavior of the agent.
        The is the basic behavior prescribed by the predefined trajectories in the commonroad scenario.

        :param delta_time: time difference to the previous step (may be needed for integration)
        """
        if self.time_step < len(self._predefined_state_list):
            # take the state defined by the scenario
            self._state = self._predefined_state_list[self.time_step]
            self._state.acceleration = 0.0
        else:
            self._valid = False

    def update_leader(self):
        """Update leader.

        This function updates the current leader, distance to it and the approaching rate.
        """
        (
            self.__leader_id,
            self.__distance_to_leader,
            self.__approaching_rate_to_leader,
        ) = self.__get_leader_commonroad()

    def update_follower(self):
        """Update follower.

        This function updates the current follower, distance to it and the approaching rate.
        """
        (
            self.__follower_id,
            self.__distance_to_follower,
            self.__approaching_rate_of_follower,
        ) = self.__get_follower_commonroad()

    def update_current_lanelet(self):
        """Update current lanelet.

        This function updates the ID current lanelet.
        """
        if (
                len(
                    self.scenario.lanelet_network.find_lanelet_by_position(
                        [self.state.position]
                    )[0]
                )
                > 0
        ):
            lanelet_id_lst = (
                self.scenario.lanelet_network.find_lanelet_by_position(
                    [self.state.position]
                )[0]
            )
            if self.expected_lanelets_list:
                retA = [i for i in lanelet_id_lst if i in self.expected_lanelets_list]
                try:
                    self.__current_lanelet_id = retA[0]
                except IndexError:
                    print('A')
                    # self.__current_lanelet_id = lanelet_id_lst[0]
                    # self.__current_lanelet_id = lanelet_id_lst[0]
            else:
                self.__current_lanelet_id = lanelet_id_lst[0]
        else:
            self.__current_lanelet_id = None

    def plot_debug(self):
        """Debug plot."""
        plot_limits = get_plot_limits_from_scenario(scenario=self.scenario)
        plt.figure(figsize=(15, 8))

        rnd = MPRenderer(draw_params={'time_begin': self.time_step}, plot_limits=plot_limits, )
        self.scenario.draw(rnd)
        rnd.render()

        # mark the ego vehicle
        rnd_ego = MPRenderer(draw_params={'time_begin': self.time_step, 'facecolor': 'r'},
                             plot_limits=plot_limits, )
        self.scenario.obstacle_by_id(self.agent_id).draw(rnd_ego)
        rnd_ego.render()

        # mark the leader vehicle
        if self.leader_id is not None:
            rnd_leader = MPRenderer(draw_params={'time_begin': self.time_step, 'facecolor': 'g'},
                                    plot_limits=plot_limits, )
            self.scenario.obstacle_by_id(self.leader_id).draw(rnd_leader)
            rnd_leader.render()

        # mark the following vehicle
        if self.follower_id is not None:
            rnd_follower = MPRenderer(draw_params={'time_begin': self.time_step, 'facecolor': 'y'},
                                      plot_limits=plot_limits, )
            self.scenario.obstacle_by_id(self.follower_id).draw(rnd_follower)
            rnd_follower.render()
        plt.title(
            'Time step: '
            + str(self.time_step)
            + '\n'
            + 'Ego ID: '
            + str(self.agent_id)
            + ' (red) \n'
            + 'Leader ID: '
            + str(self.leader_id)
            + ' (green) \n'
            + 'Follower ID: '
            + str(self.follower_id)
            + ' (yellow)'
        )

        # Obstacle IDs
        for dynamic_obstacle in self.scenario.dynamic_obstacles:
            try:
                x = dynamic_obstacle.prediction.trajectory.state_list[
                    self.time_step
                ].position[0]
                y = dynamic_obstacle.prediction.trajectory.state_list[
                    self.time_step
                ].position[1]
                plt.text(x, y, str(dynamic_obstacle.obstacle_id), zorder=100)
            except ValueError:
                pass

        plt.gca().set_aspect('equal')
        plt.show()

    def __get_leader_commonroad(self):
        """Get leader commonroad.

        Identify the leader on the current lanelet based on the current commonroad scenario

        :return: obstacle id of the next leading vehicle, distance to the leader and approaching rate
        """
        return get_leader_on_lanelet(
            scenario=self.scenario,
            ego_obstacle_id=self.agent_id,
            leader_lanelet_id=self.current_lanelet_id,
            time_step=self.time_step,
            lanelet_merged_list=self.list_lanelets_merged,
            CLCS_main=self.CLCS_main,
        )

    def __get_follower_commonroad(self):
        """Get follower commonroad.

        Identify the follower on the current lanelet based on the current commonroad scenario

        :return: obstacle id of the next following vehicle, distance to the follower and approaching rate
        """
        return get_follower_on_lanelet(
            scenario=self.scenario,
            ego_obstacle_id=self.agent_id,
            follower_lanelet_id=self.current_lanelet_id,
            time_step=self.time_step,
        )

    ####################################################################################################################
    # PROPERTIES #######################################################################################################
    ####################################################################################################################

    @property
    def scenario(self):
        """Commonroad scenario."""
        return self.__scenario

    @property
    def dt(self):
        """Time step size of the senario."""
        return self.__dt

    @property
    def agent_id(self):
        """ID of the agent."""
        return self.__agent_id

    @property
    def agent_shape(self):
        """Shape of the agent."""
        return self.__agent_shape

    @property
    def time_step(self):
        """Current time step."""
        return self._time_step

    @property
    def time(self):
        """Current time."""
        return self.__time

    @property
    def state(self):
        """State of the agent."""
        return self._state

    @property
    def initial_state(self):
        """Initial state of the agent."""
        return self.__initial_state

    @property
    def logging_enabled(self):
        """Logging enabled."""
        return self.__logging_enabled

    @property
    def log_path(self):
        """Path for logging files."""
        return self.__log_path

    @property
    def current_lanelet_id(self):
        """Current lanelet."""
        return self.__current_lanelet_id

    @property
    def debug_step(self):
        """Debug plot after every step."""
        return self.__debug_step

    @property
    def leader_id(self):
        """ID of the current leader on the same lanelet."""
        return self.__leader_id

    @property
    def distance_to_leader(self):
        """Distance to the current leader on the same lanelet."""
        return self.__distance_to_leader

    @property
    def approaching_rate_to_leader(self):
        """Approaching rate to the current leader on the same lanelet."""
        return self.__approaching_rate_to_leader

    @property
    def follower_id(self):
        """ID of the current follower on the same lanelet."""
        return self.__follower_id

    @property
    def distance_to_follower(self):
        """Distance to the current follower on the same lanelet."""
        return self.__distance_to_follower

    @property
    def approaching_rate_of_follower(self):
        """Approaching rate of the current follower on the same lanelet."""
        return self.__approaching_rate_of_follower

def clean_scenario(scenario: Scenario, agent_list: list):
    """Clean scenario.

    This functions cleans the scenario from specified agents.
    All predefined trajectories and assignments to lanelets for the agents in the list of agents are removed (except for the initial state)
    Other dynamic obstacles with IDs that are not equal to any of the agents remain.

    :param scenario: commonroad scenario
    :param agent_list: list of considered agents
    :return: new scenario with cleaned trajectories and lanelet assignments
    """
    for agent in agent_list:
        dynamic_obstacle = scenario.obstacle_by_id(agent.agent_id)
        try:
            dynamic_obstacle.prediction.trajectory.state_list = [
                dynamic_obstacle.initial_state
            ]
        except:
            pass
        # dynamic_obstacle.prediction.occupancy_set = []  # TODO: Workaround - this raises an error with commonroad io 2020.3
        dynamic_obstacle.prediction.center_lanelet_assignment = {}
        dynamic_obstacle.prediction.shape_lanelet_assignment = {}

        for lanelet in scenario.lanelet_network.lanelets:
            for t in range(1, len(agent._predefined_state_list) + 1):
                if lanelet.dynamic_obstacles_on_lanelet.get(t) is not None:
                    lanelet.dynamic_obstacles_on_lanelet.get(t).discard(agent.agent_id)

    return scenario


def update_scenario(scenario: Scenario, agent_list: list):
    """Update scenario.

    This function updates the scenario and should be called after every simulated time step.
    The new states of the specified agents are appended to the trajectories in the commonroad scenario and assigned to
    the lanelets for the corresponding time step.

    :param scenario: commonroad scenario
    :param agent_list: list of considered agents
    :return: updated scenario
    """
    for agent in agent_list:
        # create a new commonroad state
        state = State(
            position=agent.state.position,
            orientation=agent.state.orientation,
            velocity=agent.state.velocity,
            acceleration=agent.state.acceleration,
            time_step=agent.time_step,
        )

        # calculate the occupancy for the new state
        occupied_region = agent.agent_shape.rotate_translate_local(
            agent.state.position, agent.state.orientation
        )
        occupancy = Occupancy(agent.time_step, occupied_region)

        if agent.time_step == 0:
            # initial state already in scenario
            pass
        elif agent.time_step >= 1:
            # append the new state
            scenario.obstacle_by_id(
                agent.agent_id
            ).prediction.trajectory.state_list.append(state)
            scenario.obstacle_by_id(agent.agent_id).prediction.occupancy_set.append(
                occupancy
            )

        # lanelet occupancy
        if agent.current_lanelet_id is not None:
            if (
                    scenario.lanelet_network.find_lanelet_by_id(
                        agent.current_lanelet_id
                    ).dynamic_obstacles_on_lanelet.get(agent.time_step)
                    is None
            ):
                scenario.lanelet_network.find_lanelet_by_id(
                    agent.current_lanelet_id
                ).dynamic_obstacles_on_lanelet[agent.time_step] = set()
            scenario.lanelet_network.find_lanelet_by_id(
                agent.current_lanelet_id
            ).dynamic_obstacles_on_lanelet[agent.time_step].add(agent.agent_id)

    return scenario

# EOF
