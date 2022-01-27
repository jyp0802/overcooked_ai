import itertools, os
import numpy as np
import pickle, time
from overcooked_ai_py.utils import manhattan_distance
from overcooked_ai_py.planning.search import Graph, NotConnectedError
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, OvercookedGridworld, EVENT_TYPES
from overcooked_ai_py.data.planners import load_saved_action_manager, load_saved_motion_planner, PLANNERS_DIR

# Run planning logic with additional checks and
# computation to prevent or identify possible minor errors
SAFE_RUN = False

NO_COUNTERS_PARAMS = {
        'start_orientations': False,
        'wait_allowed': False,
        'counter_goals': [],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': True
}


class MotionPlanner(object):
    """A planner that computes optimal plans for a single agent to 
    arrive at goal positions and orientations in an OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
        counter_goals (list): list of positions of counters we will consider
                              as valid motion goals
    """

    def __init__(self, mdp, counter_goals=[]):
        self.mdp = mdp

        # If positions facing counters should be 
        # allowed as motion goals
        self.counter_goals = counter_goals

        # Graph problem that solves shortest path problem
        # between any position & orientation start-goal pair
        self.graph_problem = self._graph_from_grid()
        self.motion_goals_for_pos = self._get_goal_dict()

        self.all_plans = self._populate_all_plans()

    def save_to_file(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_file(filename):
        return load_saved_motion_planner(filename)

    @staticmethod
    def from_pickle_or_compute(mdp, counter_goals, custom_filename=None, force_compute=False, info=False):
        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + "_mp.pkl"

        if force_compute:
            return MotionPlanner.compute_mp(filename, mdp, counter_goals)

        try:
            mp = MotionPlanner.from_file(filename)

            if mp.counter_goals != counter_goals or mp.mdp != mdp:
                if info:
                    print("motion planner with different counter goal or mdp found, computing from scratch")
                return MotionPlanner.compute_mp(filename, mdp, counter_goals)

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            if info:
                print("Recomputing motion planner due to:", e)
            return MotionPlanner.compute_mp(filename, mdp, counter_goals)

        if info:
            print("Loaded MotionPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))
        return mp

    @staticmethod
    def compute_mp(filename, mdp, counter_goals):
        final_filepath = os.path.join(PLANNERS_DIR, filename)
        print("Computing MotionPlanner to be saved in {}".format(final_filepath))
        start_time = time.time()
        mp = MotionPlanner(mdp, counter_goals)
        print("It took {} seconds to create mp".format(time.time() - start_time))
        mp.save_to_file(final_filepath)
        return mp


    def get_plan(self, start_pos_and_or, goal_pos_and_or, end_action):
        """
        Returns pre-computed plan from initial agent position
        and orientation to a goal position and orientation.
        
        Args:
            start_pos_and_or (tuple): starting (pos, or) tuple
            goal_pos_and_or (tuple): goal (pos, or) tuple
        """
        if not end_action:
            end_action = Action.INTERACT
        plan_key = (start_pos_and_or, goal_pos_and_or, end_action)
        action_plan, pos_and_or_path, plan_cost = self.all_plans[plan_key]
        return action_plan, pos_and_or_path, plan_cost
    
    def get_gridworld_distance(self, start_pos_and_or, goal_pos_and_or):
        """Number of actions necessary to go from starting position
        and orientations to goal position and orientation (not including
        interaction action)"""
        assert self.is_valid_motion_start_goal_pair(start_pos_and_or, goal_pos_and_or), \
            "Goal position and orientation were not a valid motion goal"
        _, _, plan_cost = self.get_plan(start_pos_and_or, goal_pos_and_or, None)
        # Removing interaction cost
        return plan_cost - 1

    def _populate_all_plans(self):
        """Pre-computes all valid plans from any valid pos_or to any valid motion_goal"""
        all_plans = {}
        valid_pos_and_ors = self.mdp.get_valid_player_positions_and_orientations()
        valid_motion_goals = filter(self.is_valid_motion_goal, valid_pos_and_ors)
        for start_motion_state, goal_motion_state, end_action in itertools.product(valid_pos_and_ors, valid_motion_goals, Action.END_ACTIONS):
            if not self.is_valid_motion_start_goal_pair(start_motion_state, goal_motion_state, end_action):
                continue
            action_plan, pos_and_or_path, plan_cost = self._compute_plan(start_motion_state, goal_motion_state, end_action)
            plan_key = (start_motion_state, goal_motion_state, end_action)
            all_plans[plan_key] = (action_plan, pos_and_or_path, plan_cost)
        return all_plans

    def is_valid_motion_start_goal_pair(self, start_pos_and_or, goal_pos_and_or, end_action=None):
        if not self.is_valid_motion_goal(goal_pos_and_or, end_action):
            return False
        # the valid motion start goal needs to be in the same connected component
        if not self.positions_are_connected(start_pos_and_or, goal_pos_and_or):
            return False
        return True

    def is_valid_motion_goal(self, goal_pos_and_or, end_action=None):
        """Checks that desired single-agent goal state (position and orientation) 
        is reachable and is facing a terrain feature"""
        goal_position, goal_orientation = goal_pos_and_or
        if goal_position not in self.mdp.get_valid_player_positions():
            return False

        # Restricting goals to be facing a terrain feature
        pos_of_facing_terrain = Action.move_in_direction(goal_position, goal_orientation)
        facing_terrain_type = self.mdp.get_terrain_type_at_pos(pos_of_facing_terrain)
        # JYP: let's make it so that counters are always included for possible goals
        if facing_terrain_type == ' ':# or (facing_terrain_type == 'X' and pos_of_facing_terrain not in self.counter_goals):
            return False

        # If the end action is ACTIVATE, the faced terrain must be activatable
        if end_action == Action.ACTIVATE and facing_terrain_type not in self.mdp.get_station_terrain_names():
            return False

        return True

    def _compute_plan(self, start_motion_state, goal_motion_state, end_action):
        """Computes optimal action plan for single agent movement
        
        Args:
            start_motion_state (tuple): starting positions and orientations
            goal_motion_state (tuple): goal positions and orientations
        """
        assert self.is_valid_motion_start_goal_pair(start_motion_state, goal_motion_state, end_action)
        positions_plan = self._get_position_plan_from_graph(start_motion_state, goal_motion_state)
        return self.action_plan_from_positions(positions_plan, start_motion_state, goal_motion_state, end_action)

    def positions_are_connected(self, start_pos_and_or, goal_pos_and_or):
        return self.graph_problem.are_in_same_cc(start_pos_and_or, goal_pos_and_or)

    def _get_position_plan_from_graph(self, start_node, end_node):
        """Recovers positions to be reached by agent after the start node to reach the end node"""
        node_path = self.graph_problem.get_node_path(start_node, end_node)
        assert node_path[0] == start_node and node_path[-1] == end_node
        positions_plan = [state_node[0] for state_node in node_path[1:]]
        return positions_plan

    def action_plan_from_positions(self, position_list, start_motion_state, goal_motion_state, end_action):
        """
        Recovers an action plan that reaches the goal motion position and orientation, and 
        executes either an interact action or an activate action.
        
        Args:
            position_list (list): list of positions to be reached after the starting position
                                  (does not include starting position, but includes ending position)
            start_motion_state (tuple): starting position and orientation
            goal_motion_state (tuple): goal position and orientation

        Returns:
            action_plan (list): list of actions to reach goal state
            pos_and_or_path (list): list of (pos, or) pairs visited during plan execution
                                    (not including start, but including goal)
        """
        goal_position, goal_orientation = goal_motion_state
        action_plan, pos_and_or_path = [], []
        position_to_go = list(position_list)
        curr_pos, curr_or = start_motion_state

        # Get agent to goal position
        while position_to_go and curr_pos != goal_position:
            next_pos = position_to_go.pop(0)
            action = Action.determine_action_for_change_in_pos(curr_pos, next_pos)
            action_plan.append(action)
            curr_or = action if action != Action.STAY else curr_or
            pos_and_or_path.append((next_pos, curr_or))
            curr_pos = next_pos
        
        # Fix agent orientation if necessary
        if curr_or != goal_orientation:
            new_pos, _ = self.mdp._move_if_direction(curr_pos, curr_or, goal_orientation)
            assert new_pos == goal_position
            action_plan.append(goal_orientation)
            pos_and_or_path.append((goal_position, goal_orientation))

        # Add end action
        action_plan.append(end_action)
        pos_and_or_path.append((goal_position, goal_orientation))

        return action_plan, pos_and_or_path, len(action_plan)

    def _graph_from_grid(self):
        """Creates a graph adjacency matrix from an Overcooked MDP class."""
        state_decoder = {}
        for state_index, motion_state in enumerate(self.mdp.get_valid_player_positions_and_orientations()):
            state_decoder[state_index] = motion_state

        pos_encoder = {motion_state:state_index for state_index, motion_state in state_decoder.items()}
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for state_index, start_motion_state in state_decoder.items():
            for action, successor_motion_state in self._get_valid_successor_motion_states(start_motion_state):
                adj_pos_index = pos_encoder[successor_motion_state]
                adjacency_matrix[state_index][adj_pos_index] = self._graph_action_cost(action)

        return Graph(adjacency_matrix, pos_encoder, state_decoder)

    def _graph_action_cost(self, action):
        """Returns cost of a single-agent action"""
        assert action in Action.ALL_ACTIONS
        return 1

    def _get_valid_successor_motion_states(self, start_motion_state):
        """Get valid motion states one action away from the starting motion state."""
        start_position, start_orientation = start_motion_state
        return [(action, self.mdp._move_if_direction(start_position, start_orientation, action)) for action in Action.ALL_ACTIONS]

    def min_cost_to_feature(self, start_pos_and_or, feature_pos_list, with_argmin=False, debug=False):
        """
        Determines the minimum number of timesteps necessary for a player to go from the starting
        position and orientation to any feature in feature_pos_list and perform an interact action
        """
        start_pos = start_pos_and_or[0]
        assert self.mdp.get_terrain_type_at_pos(start_pos) != 'X'
        min_dist = np.Inf
        best_feature = None
        for feature_pos in feature_pos_list:
            for feature_goal in self.motion_goals_for_pos[feature_pos]:
                if not self.is_valid_motion_start_goal_pair(start_pos_and_or, feature_goal):
                    continue
                curr_dist = self.get_gridworld_distance(start_pos_and_or, feature_goal)
                if curr_dist < min_dist:
                    best_feature = feature_pos
                    min_dist = curr_dist
        # +1 to account for interaction action
        min_cost = min_dist + 1
        if with_argmin:
            # assert best_feature is not None, "{} vs {}".format(start_pos_and_or, feature_pos_list)
            return min_cost, best_feature
        return min_cost

    def _get_goal_dict(self):
        """Creates a dictionary of all possible goal states for all possible
        terrain features that the agent might want to interact with."""
        terrain_feature_locations = []
        for terrain_type, pos_list in self.mdp.terrain_pos_dict.items():
            if terrain_type != ' ':
                terrain_feature_locations += pos_list
        return {feature_pos:self._get_possible_motion_goals_for_feature(feature_pos) for feature_pos in terrain_feature_locations}

    def _get_possible_motion_goals_for_feature(self, goal_pos):
        """Returns a list of possible goal positions (and orientations)
        that could be used for motion planning to get to goal_pos"""
        goals = []
        valid_positions = self.mdp.get_valid_player_positions()
        for d in Direction.ALL_DIRECTIONS:
            adjacent_pos = Action.move_in_direction(goal_pos, d)
            if adjacent_pos in valid_positions:
                goal_orientation = Direction.OPPOSITE_DIRECTIONS[d]
                motion_goal = (adjacent_pos, goal_orientation)
                goals.append(motion_goal)
        return goals


class JointMotionPlanner(object):
    """A planner that computes optimal plans for a two agents to 
    arrive at goal positions and orientations in a OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
    """

    def __init__(self, mdp, params, debug=False):
        self.mdp = mdp

        # Whether starting orientations should be accounted for
        # when solving all motion problems 
        # (increases number of plans by a factor of 4)
        # but removes additional fudge factor <= 1 for each
        # joint motion plan
        self.debug = debug
        self.start_orientations = params["start_orientations"]

        # Enable both agents to have the same motion goal
        self.same_motion_goals = params["same_motion_goals"]
        
        # Single agent motion planner
        self.motion_planner = MotionPlanner(mdp, counter_goals=params["counter_goals"])

    def is_valid_jm_start_goal_pair(self, joint_start_state, joint_goal_state):
        """Checks if the combination of joint start state and joint goal state is valid"""
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        check_valid_fn = self.motion_planner.is_valid_motion_start_goal_pair
        return all([check_valid_fn(joint_start_state[i], joint_goal_state[i]) for i in range(2)])

    def is_valid_joint_motion_goal(self, joint_goal_state):
        """Checks whether the goal joint positions and orientations are a valid goal"""
        if not self.same_motion_goals and self._agents_are_in_same_position(joint_goal_state):
            return False
        multi_cc_map = len(self.motion_planner.graph_problem.connected_components) > 1
        players_in_same_cc = self.motion_planner.graph_problem.are_in_same_cc(joint_goal_state[0], joint_goal_state[1])
        if multi_cc_map and players_in_same_cc:
            return False
        return all([self.motion_planner.is_valid_motion_goal(player_state) for player_state in joint_goal_state])

    def _agents_are_in_same_position(self, joint_motion_state):
        agent_positions = [player_pos_and_or[0] for player_pos_and_or in joint_motion_state]
        return len(agent_positions) != len(set(agent_positions))


class MediumLevelActionManager(object):
    """
    Manager for medium level actions (specific joint motion goals). 
    Determines available medium level actions for each state.
    
    Args:
        mdp (OvercookedGridWorld): gridworld of interest
        mlam_params (dictionary): parameters for the medium level action manager
    """

    def __init__(self, mdp, mlam_params):
        self.mdp = mdp
        
        self.params = mlam_params
        self.wait_allowed = mlam_params['wait_allowed']
        self.counter_drop = mlam_params["counter_drop"]
        self.counter_pickup = mlam_params["counter_pickup"]
        
        self.joint_motion_planner = JointMotionPlanner(mdp, mlam_params)
        self.motion_planner = self.joint_motion_planner.motion_planner

    def save_to_file(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_file(filename):
        return load_saved_action_manager(filename)

    @staticmethod
    def from_pickle_or_compute(mdp, mlam_params, custom_filename=None, force_compute=False, info=False):
        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + "_am.pkl"

        if force_compute:
            return MediumLevelActionManager.compute_mlam(filename, mdp, mlam_params, info=info)

        try:
            mlam = MediumLevelActionManager.from_file(filename)

            if mlam.params != mlam_params or mlam.mdp != mdp:
                if info:
                    print("medium level action manager with different params or mdp found, computing from scratch")
                return MediumLevelActionManager.compute_mlam(filename, mdp, mlam_params, info=info)

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            if info:
                print("Recomputing planner due to:", e)
            return MediumLevelActionManager.compute_mlam(filename, mdp, mlam_params, info=info)

        if info:
            print("Loaded MediumLevelActionManager from {}".format(os.path.join(PLANNERS_DIR, filename)))
        return mlam

    @staticmethod
    def compute_mlam(filename, mdp, mlam_params, info=False):
        final_filepath = os.path.join(PLANNERS_DIR, filename)
        if info:
            print("Computing MediumLevelActionManager to be saved in {}".format(final_filepath))
        start_time = time.time()
        mlam = MediumLevelActionManager(mdp, mlam_params=mlam_params)
        if info:
            print("It took {} seconds to create mlam".format(time.time() - start_time))
        mlam.save_to_file(final_filepath)
        return mlam

    def pickup_ingredient_actions(self, counter_objects, ingredient):
        ingredient_dispenser_locations = self.mdp.get_terrain_locations(ingredient)
        ingredient_pickup_locations = ingredient_dispenser_locations + counter_objects[ingredient]
        return self._get_ml_actions_for_positions(ingredient_pickup_locations)

    def pickup_dish_actions(self, counter_objects, only_use_dispensers=False):
        """If only_use_dispensers is True, then only take dishes from the dispensers"""
        dish_pickup_locations = self.mdp.get_terrain_locations('dish')
        if not only_use_dispensers:
            dish_pickup_locations += counter_objects['dish']
        return self._get_ml_actions_for_positions(dish_pickup_locations)

    def pickup_counter_soup_actions(self, counter_objects):
        soup_pickup_locations = counter_objects['soup']
        return self._get_ml_actions_for_positions(soup_pickup_locations)

    def start_cooking_actions(self, pot_states_dict):
        """This is for start cooking a pot that is cookable"""
        cookable_pots_location = self.mdp.get_partially_full_pots(pot_states_dict) + \
                                 self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
        return self._get_ml_actions_for_positions(cookable_pots_location)

    def place_obj_on_counter_actions(self, state):
        all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
        valid_empty_counters = [c_pos for c_pos in self.counter_drop if c_pos in all_empty_counters]
        return self._get_ml_actions_for_positions(valid_empty_counters)

    def deliver_soup_actions(self):
        serving_locations = self.mdp.get_terrain_locations('deliver')
        return self._get_ml_actions_for_positions(serving_locations)

    def put_ingredient_in_pot_actions(self, pot_states_dict, ingredient):
        fillable_pots = self.mdp.get_partially_full_pots(pot_states_dict) + pot_states_dict['empty']
        return self._get_ml_actions_for_positions(fillable_pots)
    
    def pickup_soup_with_dish_actions(self, pot_states_dict, only_nearly_ready=False):
        ready_pot_locations = pot_states_dict['ready']
        nearly_ready_pot_locations = pot_states_dict['cooking']
        if not only_nearly_ready:
            partially_full_pots = self.mdp.get_partially_full_pots(pot_states_dict)
            nearly_ready_pot_locations = nearly_ready_pot_locations + pot_states_dict['empty'] + partially_full_pots
        return self._get_ml_actions_for_positions(ready_pot_locations + nearly_ready_pot_locations)

    def go_to_closest_feature_actions(self, player):
        feature_locations = self.mdp.get_terrain_locations('pot') + self.mdp.get_terrain_locations('dish')
        for elem in self.mdp.ALL_INGREDIENTS:
            feature_locations += self.mdp.get_terrain_locations(elem)
        closest_feature_pos = self.motion_planner.min_cost_to_feature(player.pos_and_or, feature_locations, with_argmin=True)[1]
        return self._get_ml_actions_for_positions([closest_feature_pos])

    def _get_ml_actions_for_positions(self, positions_list):
        """Determine what are the ml actions (joint motion goals) for a list of positions
        
        Args:
            positions_list (list): list of target terrain feature positions
        """
        possible_motion_goals = []
        for pos in positions_list:
            # All possible ways to reach the target feature
            for motion_goal in self.joint_motion_planner.motion_planner.motion_goals_for_pos[pos]:
                possible_motion_goals.append(motion_goal)
        return possible_motion_goals
