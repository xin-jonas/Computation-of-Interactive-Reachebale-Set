from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_route_planner.route_planner import RoutePlanner
import numpy as np
from prediction.advanced_models.behavior_sim.MOBIL.mobil import MOBILAgent
from prediction.advanced_models.behavior_sim.IDM.idm import IDMAgent
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.obstacle import DynamicObstacle
from vehiclemodels import parameters_vehicle3


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


class BaseFunctionR:
    def __init__(self, scenario, planning_problem, ego_vehicle):
        self.scenario, self.ego_vehicle, self.planning_problem = scenario, ego_vehicle, planning_problem
        self.route = None
        self.route_lanelet = None
        self.ref_CLCS = None
        self.generate_ref_route()
        self.direct_effected_vehicle = None

    def is_collision_free(self, path, sim_scenario=None):
        """
        Checks if path collides with an obstacle. Returns true for no collision and false otherwise
        :param path: The path you want to check
        """
        try:
            trajectory = Trajectory(1, path)
        except AssertionError:
            print(path[-1])
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

    def is_reached(self, test_state):
        """
        Checks if ego reach the goal area. Returns true for reached and false otherwise
        :param test_state: The state you want to check
        """
        return self.planning_problem.goal.is_reached(test_state)

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
        route = candidate_holder.retrieve_first_route()

        route = candidate_holder.retrieve_first_route()
        route_lanelet = list_routes[0].list_ids_lanelets
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
        # ego_path = [s.position for s in path]
        # time_length = len(ego_path)
        # ego_path = np.array(ego_path)
        # dist_diff = np.linalg.norm(self.route.reference_path - ego_path[0], axis=1)
        # index_min = np.argwhere(dist_diff == np.min(dist_diff))[0][0]
        # ref_path = self.route.reference_path[index_min:index_min + time_length]
        # dist = np.linalg.norm(ego_path - ref_path, axis=1)
        # # normalized_dist = dist / np.sqrt(np.sum(dist ** 2))
        # dist_sum = -np.sum(dist)
        # print(dist_sum)
        # dist_sum += -np.linalg.norm(path[-1].position - [44.5, 35])*100 # desired velocity
        proj_dist = cal_proj_distance(path[-1][0].position, path[-1][-1].position, [44.5, 35])
        dist_sum = (-proj_dist if proj_dist > 0 else 0)*100
        return dist_sum

    def cal_distance_to_obstacle(self, ego_state, time_step):
        """
        calculate the distance between ego and other vehicle
        :param ego_state: The state of the ego vehicle
        :param time_step: time step of simulation
        """
        # time_step = path[-1].time_step
        do_lst = self.scenario.dynamic_obstacles
        ego_state = np.asarray(ego_state).T
        ego_pos_sim = []
        for i in range(0, len(ego_state)):
            temp = ego_state[i]
            pos = list(np.sum([s.position for s in list(temp)], axis=0)/len(ego_state[i]))
            ego_pos_sim.append(pos)
        # distance to obstacles and other vehicles
        J_p = 0
        obj_pos = np.array([0.0, 0.0])
        for do in do_lst:
            while time_step <= ego_state[-1][0].time_step:
                if len(do.prediction.trajectory.state_list) > time_step:
                    obj_pos += do.prediction.trajectory.state_list[time_step].position
                else:
                    obj_pos += do.prediction.trajectory.state_list[-1].position
                time_step += 1
            obj_pos = obj_pos/len(ego_state)
            # print(obj_pos)
            J_p += np.mean(np.linalg.norm(ego_pos_sim - obj_pos, axis=1))
        # # print(J_p)
        J_p = 1/J_p*100
        return -J_p

    # def cal_distance_to_obstacle(self, ego_state, time_step):
    #     """
    #     calculate the distance between ego and other vehicle
    #     :param ego_state: The state of the ego vehicle
    #     :param time_step: time step of simulation
    #     """
    #     # time_step = path[-1].time_step
    #     do_lst = self.scenario.dynamic_obstacles
    #     # ego_vehicle_obs = self.scenario.obstacle_by_id(self.ego_id)
    #     ego_pos = ego_state.position
    #     # distance to obstacles and other vehicles
    #     J_p = 0
    #     for do in do_lst:
    #         if do.obstacle_id == 38:
    #             if len(do.prediction.trajectory.state_list) <= time_step:
    #                 obj_pos = do.prediction.trajectory.state_list[time_step].position
    #             else:
    #                 obj_pos = do.prediction.trajectory.state_list[-1].position
    #             # print(obj_pos)
    #             J_p += np.linalg.norm(ego_pos - obj_pos, axis=0)
    #     # # print(J_p)
    #     # J_p = J_p
    #     return J_p

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
                    'v_0': 12,
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

    def create_idm_agent_list(self):
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
                    'v_0': 12,
                    's_0': 10,
                    'T': 1,
                    'a_max': 3,
                    'a_min': -5,
                    'b': 1.5,
                    'delta': 4,
                }
                # create the MOBIL agent and append to the list
                agent_list.append(
                    IDMAgent(
                        scenario=self.scenario,
                        agent_id=dynamic_obstacle.obstacle_id,
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
        ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=search_path)
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
