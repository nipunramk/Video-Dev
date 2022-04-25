import sys
### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, 'common/')

from manim import *
import networkx as nx
import typing
from reducible_colors import *
import numpy as np
np.random.seed(23)

class MarkovChain:
    def __init__(self, states: int, edges: list[tuple[int, int]], transition_matrix=None, dist=None):
        """
        @param: states -- number of states in Markov Chain
        @param: edges -- list of tuples (u, v) for a directed edge u to v, u in range(0, states), v in range(0, states)
        @param: transition_matrix -- custom np.ndarray matrix of transition probabilities for all states in Markov chain
        @param: dist -- initial distribution across states, assumed to be uniform if none
        """
        self.states = range(states)
        self.edges = edges
        self.adj_list = {}
        for state in self.states:
            self.adj_list[state] = []
            for u, v in edges:
                if u == state:
                    self.adj_list[state].append(v)

        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            # Assume default transition matrix is uniform across all outgoing edges
            self.transition_matrix = np.zeros((states, states))
            for state in self.states:
                neighbors = self.adj_list[state]
                for neighbor in neighbors:
                    self.transition_matrix[state][neighbor] = 1 / len(neighbors)

        if dist is not None:
            self.dist = dist
        else:
            self.dist = np.array([1 / len(self.states) for _ in range(len(self.states))])

    def get_states(self):
        return list(self.states)

    def get_edges(self):
        return self.edges

    def get_adjacency_list(self):
        return self.adj_list

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_current_dist(self):
        return self.dist

    def update_dist(self):
        """
        Performs one step of the markov chain
        """
        self.dist = np.dot(self.dist, self.transition_matrix)


class MarkovChainGraph(Graph):
    def __init__(self, 
        markov_chain: MarkovChain,
        vertex_config={"stroke_color": REDUCIBLE_VIOLET, "stroke_width": 2, "fill_color": REDUCIBLE_PURPLE, "fill_opacity": 1},
        edge_config={'color': REDUCIBLE_VIOLET , 'max_tip_length_to_length_ratio': 0.1},
        **kwargs):
        super().__init__(
            markov_chain.get_states(), 
            markov_chain.get_edges(), 
            vertex_config=vertex_config,
            edge_config=edge_config,
            edge_type=Arrow, **kwargs
        )
        for edge in self.edges:
            self.scale_edge_arrow(edge)
    
    def scale_edge_arrow(self, edge: tuple[int, int]):
        u, v = edge
        arrow = self.edges[edge]
        v_c = self.vertices[v].get_center()
        u_c = self.vertices[u].get_center()
        vec = v_c - u_c
        unit_vec = vec / np.linalg.norm(vec)
        arrow_start = u_c + unit_vec * self.vertices[u].radius
        arrow_end = v_c - unit_vec * self.vertices[v].radius
        self.edges[edge] = Arrow(arrow_start, arrow_end)

class MarkovChainSimulator:
    def __init__(self, markov_chain: MarkovChain, markov_chain_g: MarkovChainGraph, num_users=50):
        self.markov_chain = markov_chain
        self.markov_chain_g = markov_chain_g
        self.num_users = num_users
        self.init_users()
    
    def init_users(self):
        self.user_to_state = {i: np.random.choice(self.markov_chain.get_states(), p=self.markov_chain.get_current_dist()) for i in range(self.num_users)}
        self.users = [Dot().set_color(REDUCIBLE_YELLOW) for _ in range(self.num_users)]
        for user_id, user in enumerate(self.users):
            user_location = self.get_user_location(user_id)
            user.move_to(user_location)

    def get_user_location(self, user: int):
        user_state = self.user_to_state[user]
        user_location = self.markov_chain_g.vertices[user_state].get_center()
        noise_v = RIGHT * np.random.normal(-0.1, 0.1) + UP * np.random.normal(-0.15, 0.15)
        return user_location + noise_v

    def get_users(self):
        return self.users

    def transition(self):
        for user_id in self.user_to_state:
            self.user_to_state[user_id] = self.update_state(user_id)

    def update_state(self, user_id: int):
        current_state = self.user_to_state[user_id]
        transition_matrix = self.markov_chain.get_transition_matrix()
        new_state = np.random.choice(self.markov_chain.get_states(), p=transition_matrix[current_state])
        return new_state

    def get_instant_transition_animations(self):
        transition_animations = []
        self.transition()
        for user_id, user in enumerate(self.users):
            new_location = self.get_user_location(user_id)
            transition_animations.append(
                user.animate.move_to(new_location)
            )
        return transition_animations

    def get_lagged_smooth_transition_animations(self):
        transition_map = {i: [] for i in self.markov_chain.get_states()}
        self.transition()
        for user_id, user in enumerate(self.users):
            new_location = self.get_user_location(user_id)
            transition_map[self.user_to_state[user_id]].append(
                user.animate.move_to(new_location)
            )
        return transition_map

class MarkovChainTester(Scene):
    def construct(self):
        markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)],
        )
        print(markov_chain.get_states())
        print(markov_chain.get_edges())
        print(markov_chain.get_current_dist())
        print(markov_chain.get_adjacency_list())
        print(markov_chain.get_transition_matrix())

        markov_chain_g = MarkovChainGraph(markov_chain, labels=True)
        self.play(
            FadeIn(markov_chain_g)
        )
        self.wait()

        markov_chain_sim = MarkovChainSimulator(markov_chain, markov_chain_g, num_users=100)
        users = markov_chain_sim.get_users()

        self.play(
            *[FadeIn(user) for user in users]
        )
        self.wait()

        num_steps = 10
        for _ in range(num_steps):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            self.play(
                *transition_animations
            )
            self.wait()

        for _ in range(num_steps):
            transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )
            self.wait()

class MarkovChainIntro(Scene):
    def construct(self):
        pass

class IntroImportanceProblem(Scene):
    def construct(self):
        pass

class IntroStationaryDistribution(Scene):
    def construct(self):
        pass


