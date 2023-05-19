from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
import numpy as np
# from prediction.advanced_models.behavior_sim.MOBIL.mobil import MOBILAgent
# from prediction.advanced_models.behavior_sim.IDM.idm import IDMAgent
from idm import IDMAgent
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.obstacle import DynamicObstacle
from vehiclemodels import parameters_vehicle3
import copy


class BaseFunctionO:
    def __init__(self, scenario, planning_problem, ego_vehicle, config):
        self.scenario, self.ego_vehicle, self.planning_problem = scenario, ego_vehicle, planning_problem
        self.route = None
        self.route_lanelet = None
        self.ref_CLCS =None
        self.generate_ref_route()
        self.direct_effected_vehicle = None
        self.desired_velocity = 10

    def is_collision_free(self, path, sim_scenario=None):
        """
        Checks if path collides with an obstacle. Returns true for no collision and false otherwise
        :param path: The path you want to check
        :param sim_scenario:
        """
        try:
            trajectory = Trajectory(1, path[1:])
        except AssertionError:
            print(path[-1])
            pass

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
            return False
        return True

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

        print(f"Number of route candidates: {num_route_candidates}")
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
        cost = - np.abs(diff_lat)*50
        # cost = 0
        if self.is_reached(path):
            return cost
        else:
            lanelets_ids = self.route_lanelet[-1]
            goal_position = self.scenario.lanelet_network.find_lanelet_by_id(lanelets_ids).center_vertices[-1]
            cost += -np.linalg.norm(path[-1].position - goal_position) * 20

            try:
                vel_interval = self.planning_problem.goal.state_list[0].velocity
                if vel_interval.start > path[-1].velocity:
                    cost += (path[-1].velocity - vel_interval.start) * 100
                elif path[-1].velocity > vel_interval.end:
                    cost += - (path[-1].velocity - vel_interval.end) * 50
            except AttributeError:
                cost += - (path[-1].velocity - self.desired_velocity) ** 2 * 50

        return cost

    def cal_distance_to_obstacle(self, ego_state, time_step, sim_scenario=None):
        """
        calculate the distance between ego and other vehicle
        :param ego_state: The state of the ego vehicle
        :param time_step: time step of simulation
        :param sim_scenario:
        """

        # time_step = path[-1].time_step
        if sim_scenario is None:
            do_lst = self.scenario.dynamic_obstacles
        else:
            do_lst = sim_scenario.dynamic_obstacles
        # ego_vehicle_obs = self.scenario.obstacle_by_id(self.ego_id)
        ego_pos = ego_state.position
        # distance to obstacles and other vehicles
        J_p = 0
        for do in do_lst:
            if len(do.prediction.trajectory.state_list) > time_step:
                obj_pos = do.prediction.trajectory.state_list[time_step].position
            else:
                continue
            # print(obj_pos)
            J_p += 1/(1e-3+np.linalg.norm(ego_pos - obj_pos, axis=0))
        # # print(J_p)
        J_p = J_p*100
        return -J_p

    def create_idm_agent_list(self, effect_vehicle=None, direct_effect_vehicle=None, undirect_effect_vehicle=None,sim_scenario=None):
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
                'v_0': 3,
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
                    'v_0': 3,
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

    def add_ego_back_scenario(self, search_path, ego_id, scenario):

        vehicle3 = parameters_vehicle3.parameters_vehicle3()
        ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)

        ego_initial_state = self.planning_problem.initial_state
        # create the planned trajectory starting at time step 0
        ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=search_path[1:])
        # create the prediction using the planned trajectory and the shape of the ego vehicle
        ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,
                                                      shape=ego_vehicle_shape)
        # the ego vehicle can be visualized by converting it into a DynamicObstacle
        ego_vehicle_type = ObstacleType.CAR
        # generate the dynamic obstacle according to the specification
        # ego_obstacle_id = scenario.generate_object_id()
        # print(ego_obstacle_id)
        ego_vehicle = DynamicObstacle(obstacle_id=ego_id, obstacle_type=ego_vehicle_type,
                                      obstacle_shape=ego_vehicle_shape, initial_state=ego_initial_state,
                                      prediction=ego_vehicle_prediction)
        # print(ego_vehicle.obstacle_shape)
        # add ego vehicle to the scenario
        scenario.add_objects(ego_vehicle)

        return scenario

    def filter_out_effected_obstacles(self, time_step, sim_scenario):

        ego_state = sim_scenario.obstacle_by_id(self.ego_vehicle.obstacle_id).prediction.trajectory.state_list[time_step]
        ego_position = ego_state.position
        ego_velocity = ego_state.velocity
        # safe distance
        safe_distance = ego_velocity*3.6/2

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
                        min_dist = dist
                        direct_effect_vehicle = dynamic_ob
                        direct_effect_vehicle_state = dynamic_ob_state
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

    def random_choose_one_action(self, state, action_list, direct_effect_vehicle_state=None):
        initial_p = [0.25, 0.25, 0.25, 0.25]
        ttc =  None
        tiv = None
        if direct_effect_vehicle_state:
            dist = np.linalg.norm(direct_effect_vehicle_state.position - state.position)
            vel_diff = state.velocity - direct_effect_vehicle_state.velocity\
                       *np.cos(direct_effect_vehicle_state.orientation - state.orientation)
            # time to close
            ttc = dist / vel_diff
            tiv = dist / state.velocity

        desired_velocity = self.desired_velocity
        # constant
        if desired_velocity*0.9 <= state.velocity <= desired_velocity*1.1:
            initial_p = [0.4, 0.25, 0.25, 0.10]
        # acc
        elif desired_velocity*0.9 > state.velocity:
            initial_p = [0.25, 0.4, 0.25, 0.10]
        # dec
        elif desired_velocity*1.1 < state.velocity <= desired_velocity*1.5:
            initial_p = [0.25, 0.10, 0.4, 0.25]
        # barking
        else:
            initial_p = [0.25, 0.10, 0.25, 0.40]

        while True:
            take_action = np.random.choice([0, 1, 2, 3], 1, p=initial_p)
            if take_action in action_list:
                return take_action

    @staticmethod
    def calc_diff_two_path(ref_path, test_path):
        ref_path = np.array(ref_path)
        ref_x = ref_path[:, 0]
        ref_y = ref_path[:, 1]
        test_path = np.array(test_path)
        test_x = test_path[:, 0]
        test_y = test_path[:, 1]
        dist_diff = np.linalg.norm(ref_path - test_path[0], axis=1)
        index_min = np.argwhere(dist_diff == np.min(dist_diff))[0][0]
        index_min_last = index_min - 1 if (index_min - 1) >= 0 else index_min
        ref_pred_y = []
        for i in range(len(test_path)):
            index_list = np.where(ref_x >= test_x[i])[0]
            index = np.where(index_list >= index_min_last)[0][0]
            ref_pred_y.append((ref_y[index]+ref_y[index+1])/2)
        ref_pred_y = np.array(ref_pred_y)
        regression_rate = np.sqrt(np.linalg.norm(ref_pred_y - test_y)/np.linalg.norm(ref_pred_y))

        return regression_rate

    def create_mobil_agent_list(self):
        # create a list of agents
        agent_list = []
        # create mobile agents depending on obstacle ID
        # marked_id = 0
        for i, dynamic_obstacle in enumerate(self.scenario.dynamic_obstacles):
            # for all dynamic obstacles with IDs in this list, a MOBIL agent will be created
            # marked_id = dynamic_obstacle.obstacle_id
            # print(marked_id)
            # if dynamic_obstacle.obstacle_id in [31]:
            # half of the agents have MOBIL behavior and desired velocity of 30 m/s ...
            if dynamic_obstacle.obstacle_id:
                idm_parameters: dict = {
                    'v_0': 15,
                    's_0': 15,
                    'T': 1,
                    'a_max': 3,
                    'a_min': -5,
                    'b': 1.5,
                    'delta': 4,
                }
                mobil_parameters: dict = {
                    'b_safe': 2,
                    'p': 0.1,
                    'a_th': 0.1,
                    'a_bias': 0.3,
                    'v_crit': 10,
                    'idm_parameters': idm_parameters,
                }

                # create the MOBIL agent and append to the list
                agent_list.append(
                    MOBILAgent(
                        scenario=self.scenario,
                        agent_id=dynamic_obstacle.obstacle_id,
                        enable_logging=False,
                        debug_step=False,
                        idm_parameters=idm_parameters,
                        mobil_parameters=mobil_parameters,
                    )
                )
        return agent_list