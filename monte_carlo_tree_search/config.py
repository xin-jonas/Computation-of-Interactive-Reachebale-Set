import logging
from pathlib import Path


class Config:
    """
    Class to collect all configurations for the MCTS.
    """
    def __init__(self):
        logging.basicConfig(level=20)

        self.SAMPLES = 10
        self.TREE_REUSE = False

        # UTC
        self.C = 0.3
        self.D = 10000

        # terminal reward
        self.TIME_PENALTY = -0.3
        self.COLLISION_PENALTY = -10000

        # visit threshold / fully expanded?
        self.VISIT_THRESHOLD = 4

        # visualization: set to true, if the solution, the expansion, the simulation or the policy tree should be
        # plotted.
        self.VISUAL = {
            'solution': False,
            'expansion': False,
            'simulation': False,
            'policy_tree': False
        }

        # multiprocessing
        self.MULTIPROCESSING = True
        self.NUM_PROCESSES = 10

        # csv file name
        self.CSV_FILE_NAME = "stats/stats.csv"
        #self.CSV_FILE_NAME = f"stats/stats_prim760_samp{self.SAMPLES}_c{self.C}_d{self.D}_" \
        #                    f"time{self.TIME_PENALTY}_coll{self.COLLISION_PENALTY}_v{self.VISIT_THRESHOLD}_s.csv"

        # scenarios
        self.PATH_SCENARIO = [x for x in Path("../../NGSIM_subset").rglob("*T-1.xml")]
        #self.PATH_SCENARIO = ["../../NGSIM_subset/Lankershim/USA_Lanker-1_1_T-1.xml",
        #                      "../../NGSIM_subset/Peachtree/USA_Peach-4_2_T-1.xml",
        #                      "../../NGSIM_subset/US101/USA_US101-26_1_T-1.xml"
        #                      ]

        # motion primitive file
        #self.FILE_MOTION_PRIMITIVES = 'V_0.0_30.0_Vstep_3.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
        #self.FILE_MOTION_PRIMITIVES = 'V_0.0_30.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
        self.FILE_MOTION_PRIMITIVES = 'V_0.0_30.0_Vstep_1.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'

        self.USE_PREP_DATA = True
