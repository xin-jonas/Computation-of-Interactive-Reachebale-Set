import numpy as np
import copy
from numpy import random
from prediction.advanced_models.agent_sim.agent import clean_scenario, update_scenario
from mcts_node import MCTSNode
from basefunctions_R import BaseFunctionO as BaseFunction
from actions_range import ActionsRange
from config import Config


class MonteCarloTreeSearchIR:
    """
    Class for action series choosing using the Single-Player Monte Carlo Tree Search.
    """

    def __init__(self, scenario, planning_problem, ego_id):
        """
        Constructor to set up the search of an action search, given by the vehicle
        id in the scenario.
        :param scenario: the CommonRoad scenario
        :param planning_problem: the specified planning problem in the scenario
        :param ego_id: the vehicle id to obtain the reference trajectory from the scenario.
        """
        self.scenario = scenario
        np.random.seed(0)
        self.config = Config()
        MCTSNode.node_count = 0
        self.ego_id = ego_id
        # remove the ego vehicle from the scenario, so that the trajectory is excluded by the collision checker.
        self.ego_vehicle = scenario.obstacle_by_id(self.ego_id)
        # scenario.remove_obstacle(self.ego_vehicle)

        # following lines necessary to avoid an exception in SearchBaseClass
        self.ego_vehicle.initial_state.yaw_rate = 0
        self.ego_vehicle.initial_state.slip_angle = 0

        # get the base initial of ego
        planning_problem.initial_state = self.ego_vehicle.initial_state
        self.trajectory = self.ego_vehicle.prediction.trajectory
        self.time_goal = len(self.trajectory.state_list)

        # BaseFunction instance
        self.Base = BaseFunction(scenario, planning_problem, self.ego_vehicle)

    def execute_search(self):
        """
        Performs the whole action selection. Coordinates the search for a sequence action until the
        final state is a terminal state.
        :return: the list of trajectory states
        """
        # interaction scenario, only speed constant, acc, dec, barking are considered
        # set agent list from scenario without ego vehicle
        agent_list = self.Base.create_idm_agent_list()
        self.scenario = clean_scenario(scenario=self.scenario, agent_list=agent_list)
        self.scenario.remove_obstacle(self.ego_vehicle)
        # monte carlo search start
        list_actions = list(np.arange(0, 4))
        root = MCTSNode(path=[[self.ego_vehicle.initial_state]], list_actions=list_actions,
                        depth_tree=0, parent=None)

        # repetition of the search for the next optimal action
        for i in range(self.time_goal):
            print('step ' + str(i))
            samples = self.config.SAMPLES
            new_root = self.one_action(root, samples)
            print("Choose node: {}".format(new_root))
            if not new_root:
                print("!!!!!!!break!!!!!!")
                # no next optimal action found (root node has no unexpanded action due to immediate collisions)
                break
            # update scenario
            # add ego vehicle back to scenario
            path = new_root.path
            path = np.array(path[1:]).T.tolist()[1]
            self.scenario = self.Base.add_ego_back_scenario(path, self.ego_id, self.scenario)
            for time_step in range(5):
                for agent in agent_list:
                    agent.step(scenario=self.scenario)
                self.scenario = update_scenario(scenario=self.scenario, agent_list=agent_list)
            self.scenario.remove_obstacle(self.ego_vehicle)

            if self.config.TREE_REUSE:
                root = copy.deepcopy(new_root)
                # root.path = new_root.path
                root.parent = None
            else:
                root = MCTSNode(path=new_root.path, list_actions=new_root.list_actions,
                                depth_tree=new_root.depth_tree, parent=root, id=0)
                if len(root.path) == root.depth_tree:
                    print('same')

            # the next optimal primitive is in a final state. The matching is finished.
            if self.is_terminal(node=root):
                print('finish simulation')
                break

        return root.path, self.scenario

    def one_action(self, root, samples):
        """
        Executes the MCTS for the next optimal action. Repeats the four main MCTS steps for a given number of
        samples.
        :param root: The root node, from where the search starts. compute budget
        :param samples: The number of repetitions of the four MCTS steps.
        :return: The next optimal action.
        """
        for i in range(samples):
            print('samples ' + str(i))
            # node = self.selection(root)
            # if node is None:
            #     # selection failed
            #     return None
            leaf = self.tree_police(root)
            # reward = self.simulation(leaf)
            if not self.is_terminal(leaf):
                print('-----------hier leaf not terminal----------')
                # print(leaf)
                reward = self.simulation(leaf)
            else:
                print('-----------hier leaf terminal----------')
                reward = self.eval_terminal_reward(leaf)
            self.backpropagation(leaf, reward)
        next_optimal_action = self.select_next_action(root)
        return next_optimal_action

    def selection(self, root):
        print('--------selection---------')
        cur_node = root
        children = cur_node.expanded_children
        max_uct = -float("inf")
        for child in children:
            uct = child.uct
            if uct >= max_uct:
                max_uct = uct
                cur_node = child
            # save the reward of the last primitive for the backpropagation
        cur_node.last_action_reward = self.eval_action_reward(cur_node.path)
        return cur_node

    def expansion(self, node):
        """
        if node has unexpanded maneuver:
            expansion with some condition within unexpanded maneuver
        elif the node is full expanded
            return random choice a node from child node list
        elif the node satisfied the terminal condition
            return current node
        """
        # path = copy.deepcopy(node.path)
        print('--------expansion---------')
        if self.is_terminal(node):
            # print('####terminal###')
            return node
        if len(node.unexpanded_actions) == 0:
            print('------full expansion------')
            # even though all unexpanded_children are expanded, the node is not marked as fully expanded due to the
            # visit threshold, returns randomly an expanded child that has less than VISIT_THRESHOLD visits
            if node.expanded_children:
                visits = np.array([c.visits < self.config.VISIT_THRESHOLD for c in node.expanded_children])
                sub_node = np.random.choice(node.expanded_children, p=visits / sum(visits))
            else:
                print('no child')
                return node
            return sub_node
        # times = 1
        # unexpanded_actions = copy.deepcopy(node.unexpanded_actions)
        while True:
            # times += 1
            if not node.unexpanded_actions:
                # print('no unexpanded_actions')
                return node

            print('actions len:' + str(len(node.unexpanded_actions)))
            print(node.unexpanded_actions)
            action_successor = np.random.choice(node.unexpanded_actions)
            node.unexpanded_actions.remove(action_successor)
            # state simulation
            path_translated = copy.deepcopy(node.path)
            for i in range(0, 5):
                path_translated = self.translate_action_to_current_state(action_successor, path_translated)
            # path_new = node.list_paths + [[node.list_paths[-1][-1]] + path_translated]
            # check collision
            collision_free = 0
            path = np.array(path_translated[1:]).T.tolist()
            for i in range(0, len(path)):
                collision_free += self.Base.is_collision_free(path[i])
            if collision_free > 1:
                print('action_successor:' + str(action_successor))
                break
            # ?????
            # calculate the reachable set for evaluation ???
            # filter out the action, which is not expected ???
            # if self.Base.is_collision_free(path_translated): # and times <= 20:
            #     # if we cant find a collision free action after 9 times random choice
            #     # we break the choice
            #     print('------new child generate ------')
            #     print('expanded-state:')
            #     print(path_translated[-1])
            #     # print('times:'+str(times))
            #     # node.unexpanded_actions.remove(action_successor)
            #     # action is collision free -> expansion successful
            #     break
            # elif times > 20:
            #     print('fail to expansion')
            #     return node
        list_actions_current = copy.copy(node.list_actions)
        # list_primitives_current.append(action_successor)
        # add expanded action to the expanded_children in the node
        print('len path translated')
        print(len(path_translated))
        node.add_child(path_translated, list_actions_current)
        node.expanded_children[-1].last_action_reward = self.eval_action_reward(path_translated)
        # print(node.expanded_children[-1])
        return node.expanded_children[-1]

    def tree_police(self, node):
        """
        while node is not terminal:
            if node has no child:
                expansion
            elif random.uniform(0,1)<.5:
                node = selection with UCT (not always expanded)
            else:
                if node is fully expanded:
                    node = selection with UCT
                else:
                    expansion
        """
        print('--------------tree_police----------')
        # cur_node = node
        while not self.is_terminal(node, collision_check=False):
            children = node.expanded_children
            if not children:
                print("------------no child----------")
                cur_node = self.expansion(node)
                return cur_node
            elif random.uniform(0, 1) < 0.2:  # node = selection with UCT (not always expanded)
                print("------------not always expand----------")
                node = self.selection(node)
            else:
                if node.fully_expanded():
                    print("------------fully expanded so selection----------")
                    node = self.selection(node)
                else:
                    print("------------expand----------")
                    cur_node = self.expansion(node)
                    return cur_node
        return node

    def simulation(self, node):
        """
        Performs the simulation step of the MCTS process following the random default policy.
        :param node: the node from where the simulation starts
        :return: the simulation reward
        """
        print('----------simulation------------')
        cur_path = copy.deepcopy(node.path)
        while not self.is_terminal(path=cur_path):
            cur_action = np.random.choice(node.list_actions)
            for i in range(0, 5):
                cur_path = self.translate_action_to_current_state(cur_action, cur_path)

        # reward for applying the action
        simulation_reward = self.eval_action_reward(cur_path)
        # compute the terminal reward for the simulation
        simulation_reward += self.eval_terminal_reward(path=cur_path)
        return simulation_reward

    def backpropagation(self, cur_node, reward):
        """
        Performs the backpropagation step of the MCTS process.
        :param cur_node: the expanded child, the backpropagation sta
        :param reward: the reward of the simulation, to be backed-up.
        """
        # logging.log(18, "Backpropagation")
        print("------------Backpropagation----------")
        # cur_node = node
        while cur_node:
            reward = cur_node.update(reward)
            # print(cur_node)
            cur_node = cur_node.parent

    def eval_action_reward(self, path):
        """
        Evaluates the reward for an applied primitive by giving a penalty for the displacement error of the
        primitive to the corresponding reference trajectory segment.
        :param path: the list states after applied action
        :return: the reward for the applied action
        """
        # ego state in last five time step
        ego_state = path[-5:]
        time_step = path[-5][0].time_step
        J_p = self.Base.cal_distance_to_obstacle(ego_state, time_step)
        # print(J_p)
        # normalize the difference distance
        J_path = self.Base.cal_diff_ref_ego_path(path)
        # print(J_path)
        return J_p + J_path

    def eval_terminal_reward(self, node: MCTSNode = None, path=None):
        """
        Evaluates the reward for a terminal state. The state is given either by a MCTSNode or by the path of the last
        applied primitive. The reward is computed by giving penalties for not reaching the time goal and for causing a
        collision.
        :param node: the terminal node, by default = None
        :param path: the path of the last applied primitive, by default = None
        :return: the reward for a terminal state
        """
        assert node is None or self.is_terminal(node)
        if node is not None:
            path = node.path
        reward = 0
        final_time = path[-1][0].time_step
        # time penalty
        if final_time < self.time_goal:
            reward += self.config.TIME_PENALTY * (self.time_goal - final_time) * self.time_goal
        # # collision penalty
        collision_free = 0
        path = np.array(path[1:]).T.tolist()
        for i in range(0, len(path)):
            collision_free += self.Base.is_collision_free(path[i])
        collision_free = True if collision_free >= 1 else False
        if not collision_free:
            reward += self.config.COLLISION_PENALTY
        return reward

    @staticmethod
    def translate_action_to_current_state(action, path, sim=False):
        if sim:
            next_state = ActionsRange(path[-1], action, 0.1, sim).take_action()
        else:
            next_state = ActionsRange(path[-1], action, 0.1).take_action()
            # print(next_state)
        path.append(next_state)
        return path

    @staticmethod
    def select_next_action(root):
        """
        Returns the next optimal primitive according to the average rewards.
        :param root: the root node
        :return: the next optimal primitive
        """
        children = root.expanded_children
        visits = np.array([c.visits for c in children])
        rewards = np.array([c.reward for c in children])
        avg_rewards = rewards / visits
        return children[np.argmax(avg_rewards)]

    def is_terminal(self, node=None, path=None, collision_check=True):
        """
        Returns whether a node or a path is terminal. Returns true if
        1. the time goal is reached by the final state,
        2. the node has no successor primitive or
        3. the primitive created a collision.
        :param node: by default = None
        :param path: by default = None
        :param collision_check: boolean if a collision check should be performed
        :return: true if the node is terminal otherwise false
        """
        if path is None:
            path = node.path
        # if node is None:
        #     no_children = False
        # else:
        #     no_children = node.no_children
        time = path[-1][0].time_step >= self.time_goal
        # collision_free = True
        # collision_free = 0
        # path = np.array(path[1:]).T.tolist()
        # for i in range(0, len(path)):
        #     collision_free += self.Base.is_collision_free(path[i])
        # collision_free = True if collision_free >= 1 else False
        # if collision_check and len(path) >= 2:
        #     collision_free = self.Base.is_collision_free(path)
        # return time or not collision_free # or no_children
        # new: use the goal region to determine whether the node is terminal
        reached = False
        if len(path) > 2:
            reached = self.Base.is_reached(path[-1][1])
        return time or reached #and not collision_free

    # def choose_one_state(self, cur_path):
    #     # start_position = (cur_path[-1].position - cur_path[0].position)/2 + cur_path[-1].position
    #     index = np.random.choice([0, 1])
    #     return cur_path[-1][index]
