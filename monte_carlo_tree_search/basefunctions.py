from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
import numpy as np
# from prediction.advanced_models.behavior_sim.MOBIL.mobil import MOBILAgent
from idm import IDMAgent
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.obstacle import DynamicObstacle
from vehiclemodels import parameters_vehicle3
from commonroad.scenario.trajectory import State
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D
import copy
import math


def cal_proj_distance(A, B, C):
    """
    :param A: left position
    :param B: right position
    :param C: goal position
    calculate the projection of AC on AB
    """
    y = np.array(B)-np.array(A)
    x = np.array(C)-np.array(A)
    distance = np.dot(x, y) / np.linalg.norm(y)
    return distance


class BaseFunction:
    def __init__(self, scenario, planning_problem, ego_vehicle):
        self.scenario, self.ego_vehicle, self.planning_problem = scenario, ego_vehicle, planning_problem
        self.route = None
        self.route_lanelet = None
        self.ref_CLCS = None
        self.generate_ref_route()
        self.direct_effected_vehicle = None
        self.generate_ref_route()
        self.desired_velocity = 5 # ego_vehicle
        self.desired_velocity_undirected = None
        self.desired_velocity_directed = None

    def is_collision_free(self, path, sim_scenario=None):
        """
        Checks if path collides with an obstacle. Returns true for no collision and false otherwise
        :param path: The path you want to check
        :param sim_scenario:
        """
        # state = path[0]
        # print('a')
        trajectory = Trajectory(path[1].time_step, path[1:])
        # create a TrajectoryPrediction object consisting of the trajectory and the shape of the ego vehicle
        traj_pred = TrajectoryPrediction(trajectory=trajectory, shape=self.ego_vehicle.obstacle_shape)

        # create a collision object using the trajectory prediction of the ego vehicle
        collision_object = create_collision_object(traj_pred)

        # check collision for action. an action is collision-free if none of its states at different time
        # steps is colliding
        if sim_scenario is None:
            collision_checker = create_collision_checker(self.scenario)
        else:
            collision_checker = create_collision_checker(sim_scenario)
        if collision_checker.collide(collision_object):
            return True
        return False

    def is_reached(self, path):
        """
        Checks if ego reach the goal area. Returns true for reached and false otherwise
        :param test_state: The state you want to check
        """
        reached = False
        for state in path[-5:]:
            if self.planning_problem.goal.is_reached(state):
                reached = True
                break
        return reached
        # return self.planning_problem.goal.is_reached(test_state)

    def generate_ref_route(self):
        """
        generate reference path with the scenario and planning_problem
        """
        route_planner = RoutePlanner(self.scenario, self.planning_problem,
                                     backend=RoutePlanner.Backend.NETWORKX_REVERSED)

        # plan routes, and save the routes in a route candidate holder
        candidate_holder = route_planner.plan_routes()

        # option 1: retrieve all routes
        list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
        # print(f"Number of route candidates: {num_route_candidates}")
        # here we retrieve the first route in the list, this is equivalent to: route = list_routes[0]
        # route = candidate_holder.retrieve_first_route()
        route = candidate_holder.retrieve_best_route_by_orientation()
        route_lanelet = route.list_ids_lanelets
        # visualize_route(route, draw_route_lanelets=True, draw_reference_path=False, size_x=6)
        self.route = route
        self.route_lanelet = route_lanelet
        CLCS = CurvilinearCoordinateSystem(self.route.reference_path)
        CLCS.compute_and_set_curvature()
        self.ref_CLCS = CLCS

    def cal_diff_ref_ego_path(self, path):
        """
        calculate the distance difference with the generated reference path
        :param path: The path you want to compare with reference path
        """
        ego_path_pos = [s.position for s in path[-5:]]
        diff_lat = 0
        for pos in ego_path_pos:
            try:
                p_lon, p_lat = self.ref_CLCS.convert_to_curvilinear_coords(pos[0], pos[1])
                diff_lat += p_lat
            except ValueError:
                continue
        cost = - np.abs(diff_lat) * 50
        # cost = 0
        if self.is_reached(path):
            return cost
        else:
            lanelets_ids = self.route_lanelet[-1]
            goal_position = self.scenario.lanelet_network.find_lanelet_by_id(lanelets_ids).center_vertices[-1]
            try:
                p_lon_diff, _ = self.ref_CLCS.convert_to_curvilinear_coords(goal_position[0], goal_position[1]) - \
                        self.ref_CLCS.convert_to_curvilinear_coords(path[-1].position[0], path[-1].position[1])
                if p_lon_diff < 0 or p_lon_diff > 100:
                    p_lon_diff = 100
                cost += -p_lon_diff*20
            except ValueError:
                dist = np.linalg.norm(path[-1].position - goal_position)
                if dist > 100:
                    dist = 100
                cost += -dist * 20

            try:
                vel_interval = self.planning_problem.goal.state_list[0].velocity
                if vel_interval.start > path[-1].velocity:
                    cost += (path[-1].velocity - vel_interval.start) * 100
                elif path[-1].velocity > vel_interval.end:
                    cost += - (path[-1].velocity - vel_interval.end) * 50
            except AttributeError:
                cost += - (path[-1].velocity - self.desired_velocity)**2 * 50
        return cost

    def cal_distance_to_obstacle(self, ego_state, time_step, sim_scenario=None):
        """
        calculate the distance between ego and other vehicle
        :param ego_state: The state of the ego vehicle
        :param time_step: time step of simulation
        :param sim_scenario:
        """

        # time_step = path[-1].time_step
        j_p_d = 0
        ego_pos = ego_state.position
        if sim_scenario is None:
            do_lst = self.scenario.dynamic_obstacles
        else:
            do_lst = sim_scenario.dynamic_obstacles
        # ego_vehicle_obs = self.scenario.obstacle_by_id(self.ego_id)
        # distance to obstacles and other vehicles
        j_p = 0
        for do in do_lst:
            if len(do.prediction.trajectory.state_list) > time_step and do.obstacle_id != self.ego_vehicle.obstacle_id:
                obj_pos = do.prediction.trajectory.state_list[time_step].position
                if self.direct_effected_vehicle is not None and do.obstacle_id == self.direct_effected_vehicle:
                    obj_pos = do.prediction.trajectory.state_list[time_step].position
                    j_p_d += -1 / (1e-4 + np.linalg.norm(ego_pos - obj_pos, axis=0)) * 1600
                    continue
            else:
                continue
            # print(obj_pos)
            j_p += 1 / (1e-4 + np.linalg.norm(ego_pos - obj_pos, axis=0))
        j_p = -j_p * 400
        return j_p + j_p_d

    def create_idm_agent_list(self, effect_vehicle=None, direct_effect_vehicle=None, undirect_effect_vehicle=None,
                              sim_scenario=None):
        # create a list of agents
        agent_list = []
        # create mobile agents depending on obstacle ID
        # tinydict = {31: [8, 6, 2], 38: [10, 4, 2], 39: [16, 14, 7]}
        tinydict = None
        if undirect_effect_vehicle:
            for dynamic_obstacle_id in undirect_effect_vehicle:
                idm_parameters: dict = {
                    'v_0': 8,
                    's_0': 10,
                    'T': 1,
                    'a_max': 3,
                    'a_min': -5,
                    'b': 1.5,
                    'delta': 4,
                    'label': 3,
                    'action': None,
                }
                # create the MOBIL agent and append to the list
                agent_list.append(
                    IDMAgent(
                        scenario=sim_scenario if sim_scenario else self.scenario,
                        agent_id=dynamic_obstacle_id,
                        expected_lanelets_list=tinydict[dynamic_obstacle_id] if tinydict else None,
                        enable_logging=False,
                        debug_step=False,
                        idm_parameters=idm_parameters,
                    )
                )
        # direct effect vehicle
        if direct_effect_vehicle:
            idm_parameters: dict = {
                'v_0': 15,
                's_0': 10,
                'T': 1,
                'a_max': 3,
                'a_min': -5,
                'b': 1.5,
                'delta': 4,
                'label': 1,
                'action': None,
            }
            # create the Idm agent and append to the list
            agent_list.append(
                IDMAgent(
                    scenario=sim_scenario if sim_scenario else self.scenario,
                    agent_id=direct_effect_vehicle,
                    expected_lanelets_list=tinydict[direct_effect_vehicle] if tinydict else None,
                    enable_logging=False,
                    debug_step=False,
                    idm_parameters=idm_parameters,
                )
            )
        # ego vehicle
        idm_parameters: dict = {
            'v_0': self.desired_velocity,
            's_0': 10,
            'T': 1,
            'a_max': 7,
            'a_min': -6,
            'b': 1.5,
            'delta': 4,
            'label': 0,
            'action': None,
        }
        agent_list.append(
            IDMAgent(
                scenario=sim_scenario if sim_scenario else self.scenario,
                agent_id=self.ego_vehicle.obstacle_id,
                expected_lanelets_list=self.route_lanelet,
                ref_CLCS=self.ref_CLCS,
                enable_logging=False,
                debug_step=False,
                idm_parameters=idm_parameters,
            )
        )
        if effect_vehicle:
            for dynamic_obstacle_id in effect_vehicle:
                idm_parameters: dict = {
                    'v_0': 15,
                    's_0': 10,
                    'T': 1,
                    'a_max': 3,
                    'a_min': -5,
                    'b': 1.5,
                    'delta': 4,
                    'label': 2,
                    'action': None,
                }
                # create the MOBIL agent and append to the list
                agent_list.append(
                    IDMAgent(
                        scenario=sim_scenario if sim_scenario else self.scenario,
                        agent_id=dynamic_obstacle_id,
                        expected_lanelets_list=tinydict[dynamic_obstacle_id] if tinydict else None,
                        enable_logging=False,
                        debug_step=False,
                        idm_parameters=idm_parameters,
                    )
                )

        return agent_list

    def random_choose_one_action(self, state, action_list, direct_effect_vehicle_state=None):
        initial_p = [0.25, 0.25, 0.25, 0.25, 0.25]
        risk_rank = None
        if direct_effect_vehicle_state is not None:
            try:
                dist, _ = self.ref_CLCS.convert_to_curvilinear_coords(\
                    direct_effect_vehicle_state.position[0], direct_effect_vehicle_state.position[1]) - \
                       self.ref_CLCS.convert_to_curvilinear_coords(state.position[0], state.position[1])
                vel_diff = state.velocity - direct_effect_vehicle_state.velocity\
                           *np.cos(direct_effect_vehicle_state.orientation - state.orientation)
                # time to close
                v_ego = state.velocity
                v_leader = direct_effect_vehicle_state.velocity\
                           *np.cos(direct_effect_vehicle_state.orientation - state.orientation)
                risk_rank = self.action_risk_rank(dist, vel_diff, v_ego, v_leader)
            except:
                risk_rank = None
        desired_velocity = self.desired_velocity

        # constant
        if desired_velocity*0.9 <= state.velocity <= desired_velocity*1.1:
            initial_p = [0.1, 0.25, 0.3, 0.25, 0.10]
        # acc
        elif desired_velocity*0.9 > state.velocity > desired_velocity*0.5:
            initial_p = [0.15, 0.15, 0.15, 0.3, 0.25]
        # acc
        elif state.velocity <= desired_velocity*0.5:
            initial_p = [0.15, 0.15, 0.15, 0.25, 0.3]
        # dec
        elif desired_velocity*1.1 < state.velocity <= desired_velocity*1.5:
            initial_p = [0.25, 0.3, 0.15, 0.15, 0.15]
        # barking
        else:
            initial_p = [0.3, 0.25, 0.15, 0.15, 0.15]

        while True:
            if risk_rank is not None:
                p = np.sum([initial_p, risk_rank], axis=0).tolist()
                take_action = np.random.choice([-2, -1, 0, 1, 2], 1, p=[e/2 for e in p])
            else:
                take_action = np.random.choice([-2, -1, 0, 1, 2], 1, p=initial_p)
            if take_action in action_list:
                return take_action

    def add_ego_back_scenario(self, search_path, ego_id, scenario):
        vehicle3 = parameters_vehicle3.parameters_vehicle3()
        ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)

        ego_initial_state = self.planning_problem.initial_state
        # create the planned trajectory starting at time step 0
        ego_vehicle_trajectory = Trajectory(initial_time_step=search_path[0].time_step, state_list=search_path)
        # create the prediction using the planned trajectory and the shape of the ego vehicle
        ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,
                                                      shape=ego_vehicle_shape)
        # the ego vehicle can be visualized by converting it into a DynamicObstacle
        ego_vehicle_type = ObstacleType.CAR
        # generate the dynamic obstacle according to the specification
        ego_vehicle = DynamicObstacle(obstacle_id=ego_id, obstacle_type=ego_vehicle_type,
                                      obstacle_shape=ego_vehicle_shape, initial_state=ego_initial_state,
                                      prediction=ego_vehicle_prediction)
        # print(ego_vehicle.obstacle_shape)
        # add ego vehicle to the scenario
        scenario.add_objects(ego_vehicle)
        return scenario

    def sample_path(self, state_list, sample_nums, action):
        # acc_range= [state_list[0].acceleration, state_list[-1].acceleration]
        vel_range = [state_list[0].velocity, state_list[-1].velocity]
        center_points = np.array(self.ref_CLCS.reference_path())
        try:
            ego_lanelet_spline = CubicSpline2D(center_points[:, 0], center_points[:, 1])
        except ValueError:
            center_points = np.unique(center_points, axis=0)
            ego_lanelet_spline = CubicSpline2D(center_points[:, 0], center_points[:, 1])

        acc_range = self.action_system(action)
        num = 10
        if sample_nums >= num:
            sample_nums = num - 1

        acc_samples = np.random.choice(np.linspace(acc_range[0], acc_range[-1], num=num), sample_nums, replace=False)
        vel_sample = np.random.choice(np.linspace(vel_range[0], vel_range[-1], num=num), 1)[0]
        # a_sample - a_min
        acc_diff_to_min = (vel_sample - vel_range[0])/self.scenario.dt
        # calculate the minimum position (arc length) travelled along the spline
        s_min = ego_lanelet_spline.get_min_arc_length(state_list[0].position)[0]
        s_sample = s_min + 0.5*acc_diff_to_min*self.scenario.dt**2
        # pos_sample = np.random.choice(np.linspace(acc_range[0], acc_range[-1], num=10), 1)
        path_list = []
        for acc in acc_samples:
            vel_start = vel_sample
            s_start = s_sample
            path = []
            for i in range(1, 6):
                vel_start = vel_start + acc * self.scenario.dt
                if vel_start < 0:
                    acc = - vel_start / self.scenario.dt
                # covered distance along the center line of the current lanelet
                ds = vel_start * self.scenario.dt + 1 / 2 * acc * self.scenario.dt ** 2
                s_new = s_start + ds

                # new position
                x, y = ego_lanelet_spline.calc_position(s_new)
                position = np.array([x, y])

                # new orientation
                orientation = ego_lanelet_spline.calc_yaw(s_new)
                state = State(
                    position=position,
                    orientation=orientation,
                    velocity=vel_start,
                    acceleration=acc,
                    time_step=state_list[0].time_step + i,
                )
                path.append(state)

            path_list.append(path)
        return path_list

    def filter_out_effected_obstacles(self, time_step, sim_scenario):
        try:
            ego_state = sim_scenario.obstacle_by_id(self.ego_vehicle.obstacle_id).prediction.trajectory.state_list[0]
        except IndexError:
            print('')
        ego_position = ego_state.position
        p_lon_ego, _ = self.ref_CLCS.convert_to_curvilinear_coords(ego_position[0], ego_position[1])
        ego_velocity = ego_state.velocity
        # safe distance
        safe_distance = ego_velocity*3.6/2 + self.ego_vehicle.obstacle_shape.length

        effect_vehicle = []
        direct_effect_vehicle = None
        direct_effect_vehicle_state = None
        undirect_effect_vehicle = []
        min_dist = 1000

        dynamic_obstacle = sim_scenario.dynamic_obstacles
        # get vehicle which effects ego or is effected by ego
        for dynamic_ob in dynamic_obstacle:
            if dynamic_ob.obstacle_id != self.ego_vehicle.obstacle_id:
                try:
                    dynamic_ob_state = dynamic_ob.prediction.trajectory.state_list[time_step]
                except IndexError:
                    continue
                    # dynamic_ob_state = dynamic_ob.prediction.trajectory.state_list[-1]

                dynamic_ob_position = dynamic_ob_state.position
                dist = np.linalg.norm(dynamic_ob_position-ego_position)
                if dist <= safe_distance:
                    lanelet_id_lst = self.scenario.lanelet_network.find_lanelet_by_position([dynamic_ob_position])[0]
                    if len(set(lanelet_id_lst).intersection(set(self.route_lanelet))) > 0 and min_dist > dist:
                        p_lon_vehicle, _ = self.ref_CLCS.convert_to_curvilinear_coords(dynamic_ob_position[0], dynamic_ob_position[1])
                        if p_lon_vehicle - p_lon_ego > 0:
                            min_dist = dist
                            direct_effect_vehicle = dynamic_ob
                            direct_effect_vehicle_state = dynamic_ob_state
                        else:
                            effect_vehicle.append(dynamic_ob)
                        # min_dist = dist
                        # direct_effect_vehicle = dynamic_ob
                        # direct_effect_vehicle_state = dynamic_ob_state
                    else:
                        effect_vehicle.append(dynamic_ob)
                    dynamic_ob.initial_state = copy.deepcopy(dynamic_ob_state)
                    dynamic_ob.initial_state.time_step = 0

        # undirected effect vehicles
        if direct_effect_vehicle:
            direct_effect_vehicle_pos = direct_effect_vehicle_state.position
            safe_distance = direct_effect_vehicle_state.velocity*1.8
            for dynamic_ob in dynamic_obstacle:
                if dynamic_ob.obstacle_id != self.ego_vehicle.obstacle_id \
                        and dynamic_ob.obstacle_id != direct_effect_vehicle.obstacle_id:
                    try:
                        dynamic_ob_state = dynamic_ob.prediction.trajectory.state_list[time_step]
                    except IndexError:
                        dynamic_ob_state = dynamic_ob.prediction.trajectory.state_list[-1]

                    dynamic_ob_position = dynamic_ob_state.position
                    dist = np.linalg.norm((dynamic_ob_position - direct_effect_vehicle_pos))
                    if dist <= safe_distance:
                        undirect_effect_vehicle.append(dynamic_ob)
                        dynamic_ob.initial_state = copy.deepcopy(dynamic_ob_state)
                        dynamic_ob.initial_state.time_step = 0
        # if vehicle is undirected effect vehicle, the cant be effect vehicle
        effect_vehicle = list(set(effect_vehicle).difference(set(undirect_effect_vehicle)))

        for dynamic_ob in dynamic_obstacle:
            if dynamic_ob not in effect_vehicle and dynamic_ob not in undirect_effect_vehicle \
                    and dynamic_ob != direct_effect_vehicle and dynamic_ob.obstacle_id != self.ego_vehicle.obstacle_id:
                sim_scenario.remove_obstacle(dynamic_ob)

        sim_scenario.obstacle_by_id(self.ego_vehicle.obstacle_id).initial_state = copy.deepcopy(ego_state)
        sim_scenario.obstacle_by_id(self.ego_vehicle.obstacle_id).initial_state.time_step = 0

        effect_vehicle = [ob.obstacle_id for ob in effect_vehicle]
        if direct_effect_vehicle:
            direct_effect_vehicle = direct_effect_vehicle.obstacle_id
            self.direct_effected_vehicle = direct_effect_vehicle
        else:
            direct_effect_vehicle = None
        undirect_effect_vehicle = [ob.obstacle_id for ob in undirect_effect_vehicle]

        return effect_vehicle, direct_effect_vehicle, undirect_effect_vehicle, sim_scenario

    @staticmethod
    def calculate_reward_sim(path_sim_result):
        # sim_result = {
        #     'path_id': 0,
        #     'is_terminal': 0,
        #     'simulation_reward': 0,
        #     'terminal_reward': 0,
        #     'sum_reward': 0,
        # }
        collision_free = 0
        sim_reward_c = 0
        ter_reward_c = 0
        sim_reward = 0
        ter_reward = 0
        for i, sim_result in enumerate(path_sim_result):

            if sim_result['is_terminal'] == 1:
                collision_free += 1
                ter_reward_c += sim_result['terminal_reward']
                sim_reward_c += sim_result['simulation_reward']
            else:
                ter_reward += sim_result['terminal_reward']
                sim_reward += sim_result['simulation_reward']

        if collision_free/len(path_sim_result) > 0.6:
            return (ter_reward_c + sim_reward_c)/collision_free
        else:
            return (ter_reward + sim_reward)/(len(path_sim_result) - collision_free)

    @staticmethod
    def action_system(action):
        system_num = 2
        acc_range = None
        if system_num == 0:
            if action == 0:
                acc_range = [-1, 1]
            elif action == 1:
                acc_range = [1, 7]
            elif action == -1:
                acc_range = [-3, -1]
            elif action == -2:
                acc_range = [-6, -3]
        elif system_num == 1:
            if action == 0:
                acc_range = [-1, 1]
            elif action == 1:
                acc_range = [1, 4]
            elif action == 2:
                acc_range = [4, 7]
            elif action == -1:
                acc_range = [-3, -1]
            elif action == -2:
                acc_range = [-6, -3]
        elif system_num == 2:
            if action == 0:
                acc_range = [-2, 2]
            elif action == 1:
                acc_range = [1, 5]
            elif action == 2:
                acc_range = [4, 7]
            elif action == -1:
                acc_range = [-4, -1]
            elif action == -2:
                acc_range = [-6, -3]
        elif system_num == 3:
            if action == 0:
                acc_range = [-1, 1]
            elif action == 1:
                acc_range = [0.5, 2.5]
            elif action == 2:
                acc_range = [2, 4.5]
            elif action == 3:
                acc_range = [4, 7]
            elif action == -1:
                acc_range = [-2.5, -0.5]
            elif action == -2:
                acc_range = [-4.5, -2]
            elif action == -3:
                acc_range = [-6, -4]
        return acc_range

    @staticmethod
    def ttc_pdf(t):
        if 0 <= t <= 1:
            return 1
        elif t >= 10:
            return 0
        else:
            return -1/9*t + 10/9

    @staticmethod
    def tiv_pdf(t):
        if 0 <= t <= 0.5:
            return 1
        elif t <= 1:
            return -t + 1.5
        elif t <= 2:
            return -0.5*t + 1
        else:
            return 0

    @staticmethod
    def ees_cal(v_ego, v_leader):
        return 1.2 * (v_ego - v_leader)

    def risk_eval(self, ttc, tiv, v_ego, v_leader):
        ees_tiv = max(self.ees_cal(v_ego, v_leader), self.ees_cal(v_ego, v_leader - 6*tiv))
        ees_ttc = self.ees_cal(v_ego, v_leader)
        r_ttc = self.ttc_pdf(ttc)*ees_ttc
        r_tiv = self.ttc_pdf(tiv)*ees_tiv
        return r_ttc + r_tiv

    @staticmethod
    def ttc_tiv_cal(dist, v_diff, v_ego, acc):
        dist_new = dist + v_diff*0.5 - 0.5*acc*0.5**2
        v_diff_new = v_diff - acc*0.5
        ttc = dist_new / v_diff_new
        v_ego_new = v_ego + 0.5*acc
        tiv = dist / v_ego_new
        return ttc, tiv

    def action_risk_rank(self, dist, v_diff, v_ego, v_leader):
        risk_list = []
        for action in range(-2, 3):
            acc = self.action_system(action)
            acc_sample = np.random.choice(np.linspace(acc[0], acc[-1], num=10), 6, replace=False)

            risk = 0
            for acc in acc_sample:
                ttc, tiv = self.ttc_tiv_cal(dist, v_diff, v_ego, acc)
                risk += self.risk_eval(ttc, tiv, v_ego, v_leader)
            risk_list.append(risk/6)
        sorted_id = sorted(range(len(risk_list)), key=lambda k: risk_list[k], reverse=False)
        for i, id in enumerate(sorted_id):
            risk_list[id] = i
        risk_rank = [risk / sum(risk_list) for risk in risk_list]
        return risk_rank

