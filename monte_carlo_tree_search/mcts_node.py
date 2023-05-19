import copy
import numpy as np
from config import Config


class MCTSNode:
    """
    Class for nodes used in Monte Carlo Tree Search.
    """
    node_count = 0
    config = Config()

    @classmethod
    def get_node_id(cls):
        count = cls.node_count
        cls.node_count += 1
        return count

    def __init__(self, path, list_actions, parent, depth_tree, id=None):
        """
        Constructor for a MCTSNode
        :param path: sate list of trajectory of car
        :param list_actions: list of usable actions
        :param depth_tree: tree depth of the node
        :param parent: the parent node
        :param id: default=None
        """
        # super().__init__(path, list_actions, depth_tree)
        self.list_actions = list_actions
        self.path = path
        self.take_action = None
        self.expanded_children = []
        self.unexpanded_actions = copy.deepcopy(list_actions)
        self.parent = parent
        self.visits = 0.0
        self.reward = 0.0
        self.reward_squared = 0.0
        # the reward of the last applied action, cache to compute the reward in the backpropagation
        self.last_action_reward = 0.0
        self.depth_tree = depth_tree
        if id is None:
            self.id = self.get_node_id()
        else:
            self.id = id

    def update(self, reward):
        """
        Updates the accumulated rewards and the visits of the node during the backpropagation
        :param reward: to be added to the accumulated rewards
        :return: the reward added by the last action reward
        """
        reward = reward + self.last_action_reward
        self.reward += reward
        self.reward_squared += reward ** 2
        self.visits += 1
        return reward

    def add_child(self, path, list_actions, id=None, action=None):
        """
        Adds a child to the expanded_children list
        :param path: path of ego
        :param action: parent take "action" to child
        :param list_actions: list of action
        :param id: default=None
        :return: the expanded child and unexpanded_actions
        """
        child = MCTSNode(path=path, list_actions=list_actions, depth_tree=self.depth_tree + 1,
                         parent=self, id=id)
        self.expanded_children.append(child)
        if action is not None:
            child.take_action = action
        # self.unexpanded_actions.remove(action)

    def fully_expanded(self):
        """
        Returns whether the node is fully expanded.
        Returns true if no unexpanded action is available
        and every expanded child has been visited at least VISIT_THRESHOLD times,
        otherwise false.
        :return: true if fully expanded otherwise false
        """
        return len(self.unexpanded_actions) == 0 and \
               all([c.visits >= self.config.VISIT_THRESHOLD for c in self.expanded_children])

    @property
    def no_children(self):
        """
        returns whether the action successors list is empty
        :return: true for empty otherwise false
        """
        return len(self.expanded_children) == 0

    @property
    def uct(self):
        """
        Returns the uct value according to the single-player uct formulation.
        :return: uct value
        """
        c = self.config.C
        w = self.reward
        n = self.visits if (self.visits != 0) else 1
        s = self.reward_squared
        t = self.parent.visits if (self.parent and self.parent.visits != 0) else 1
        d = self.config.D
        # exploitation = max(min(((w / n) + 5000) / 5000 * 2 - 1, 1.0), -1.0)
        # exploitation = 3 * np.exp(0.02 * (w / n))
        exploitation = w / n
        exploration = np.sqrt(np.log(t) / n)
        modification = np.sqrt((s - n * (w / n) ** 2 + d) / n)
        uct = exploitation + c * exploration + modification
        return uct

    def __repr__(self):
        return f"Id: {self.id}; Cost: {self.reward}; Visits: {self.visits}; UCT: {self.uct}; " \
               f"Fully explored?: {self.fully_expanded()}; Leaf?: {len(self.expanded_children) == 0}; Path: " \
               f"{len(self.path)}; depth_tree:{self.depth_tree};children:{len(self.expanded_children)}"

