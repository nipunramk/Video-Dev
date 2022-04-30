import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from reducible_colors import *

from typing import Hashable

import numpy as np
import itertools as it

np.random.seed(23)


class MarkovChain:
    def __init__(
        self,
        states: int,
        edges: list[tuple[int, int]],
        transition_matrix=None,
        dist=None,
    ):
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
            self.dist = np.array(
                [1 / len(self.states) for _ in range(len(self.states))]
            )

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

    def get_true_stationary_dist(self):
        dist = np.linalg.eig(np.transpose(self.transition_matrix))[1][:, 0]
        return dist / sum(dist)


class CustomLabel(Text):
    def __init__(self, label, font="SF Mono", scale=1, weight=BOLD):
        super().__init__(label, font=font, weight=weight)
        self.scale(scale)


class CustomCurvedArrow(CurvedArrow):
    def __init__(self, start, end, tip_length=0.15, **kwargs):
        self.start = start
        self.end = end

        super().__init__(start, end, **kwargs)
        self.pop_tips()
        self.add_tip(
            tip_shape=ArrowTriangleFilledTip,
            tip_length=tip_length,
            at_start=False,
        )
        self.tip.z_index = -100


class MarkovChainGraph(Graph):
    def __init__(
        self,
        markov_chain: MarkovChain,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE,
            "fill_opacity": 0.5,
        },
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        enable_curved_double_arrows=True,
        labels=True,
        **kwargs,
    ):
        self.markov_chain = markov_chain
        self.enable_curved_double_arrows = enable_curved_double_arrows

        self.default_curved_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "stroke_width": 3,
            "radius": 4,
        }

        self.default_straight_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "max_tip_length_to_length_ratio": 0.06,
            "stroke_width": 3,
        }

        if labels:
            labels = {
                k: CustomLabel(str(k), scale=0.6) for k in markov_chain.get_states()
            }

        self.labels = []

        super().__init__(
            markov_chain.get_states(),
            markov_chain.get_edges(),
            vertex_config=vertex_config,
            labels=labels,
            **kwargs,
        )

        self._graph = self._graph.to_directed()
        self.remove_edges(*self.edges)

        self.add_markov_chain_edges(
            *markov_chain.get_edges(),
            straight_edge_config=straight_edge_config,
            curved_edge_config=curved_edge_config,
        )

        self.clear_updaters()

        # this updater makes sure the edges remain connected
        # even when states move around
        def update_edges(graph):
            for (u, v), edge in graph.edges.items():
                v_c = self.vertices[v].get_center()
                u_c = self.vertices[u].get_center()
                vec = v_c - u_c
                unit_vec = vec / np.linalg.norm(vec)

                arrow_start = u_c + unit_vec * (self.vertices[u].width / 2)
                arrow_end = v_c - unit_vec * (self.vertices[v].width / 2)

                edge.put_start_and_end_on(arrow_start, arrow_end)

        self.add_updater(update_edges, 0)

    def add_edge_buff(
        self,
        edge: tuple[Hashable, Hashable],
        edge_type: type[Mobject] = None,
        edge_config: dict = None,
    ):
        """
        Custom function to add edges to our Markov Chain,
        making sure the arrowheads land properly on the states.
        """
        if edge_config is None:
            edge_config = self.default_edge_config.copy()
        added_mobjects = []
        for v in edge:
            if v not in self.vertices:
                added_mobjects.append(self._add_vertex(v))
        u, v = edge

        self._graph.add_edge(u, v)

        base_edge_config = self.default_edge_config.copy()
        base_edge_config.update(edge_config)
        edge_config = base_edge_config
        self._edge_config[(u, v)] = edge_config

        v_c = self.vertices[v].get_center()
        u_c = self.vertices[u].get_center()
        vec = v_c - u_c
        unit_vec = vec / np.linalg.norm(vec)

        if self.enable_curved_double_arrows:
            arrow_start = u_c + unit_vec * (self.vertices[u].width / 2)
            arrow_end = v_c - unit_vec * (self.vertices[v].width / 2)
        else:
            arrow_start = u_c
            arrow_end = v_c
            edge_config["buff"] = self.vertices[u].radius

        edge_mobject = edge_type(
            start=arrow_start, end=arrow_end, z_index=-100, **edge_config
        )
        self.edges[(u, v)] = edge_mobject

        self.add(edge_mobject)
        added_mobjects.append(edge_mobject)
        return self.get_group_class()(*added_mobjects)

    def add_markov_chain_edges(
        self,
        *edges: tuple[Hashable, Hashable],
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        **kwargs,
    ):
        """
        Custom function for our specific case of Markov Chains.
        This function aims to make double arrows curved when two nodes
        point to each other, leaving the other ones straight.
        Parameters
        ----------
        - edges: a list of tuples connecting states of the Markov Chain
        - curved_edge_config: a dictionary specifying the configuration
        for CurvedArrows, if any
        - straight_edge_config: a dictionary specifying the configuration
        for Arrows
        """

        if curved_edge_config is not None:
            curved_config_copy = self.default_curved_edge_config.copy()
            curved_config_copy.update(curved_edge_config)
            curved_edge_config = curved_config_copy
        else:
            curved_edge_config = self.default_curved_edge_config.copy()

        if straight_edge_config is not None:
            straight_config_copy = self.default_straight_edge_config.copy()
            straight_config_copy.update(straight_edge_config)
            straight_edge_config = straight_config_copy
        else:
            straight_edge_config = self.default_straight_edge_config.copy()

        edge_vertices = set(it.chain(*edges))
        new_vertices = [v for v in edge_vertices if v not in self.vertices]
        added_vertices = self.add_vertices(*new_vertices, **kwargs)

        edge_types_dict = {}
        for e in edges:
            if self.enable_curved_double_arrows and (e[1], e[0]) in edges:
                edge_types_dict.update({e: (CustomCurvedArrow, curved_edge_config)})

            else:
                edge_types_dict.update({e: (Arrow, straight_edge_config)})

        added_mobjects = sum(
            (
                self.add_edge_buff(
                    edge,
                    edge_type=e_type_and_config[0],
                    edge_config=e_type_and_config[1],
                ).submobjects
                for edge, e_type_and_config in edge_types_dict.items()
            ),
            added_vertices,
        )

        return self.get_group_class()(*added_mobjects)

    def get_transition_labels(self):
        """
        This function returns a VGroup with the probability that each
        each state has to transition to another state, based on the
        Chain's transition matrix.
        It essentially takes each edge's probability and creates a label to put
        on top of it, for easier indication and explanation.
        This function returns the labels already set up in a VGroup, ready to just
        be created.
        """
        tm = self.markov_chain.get_transition_matrix()

        labels = VGroup()
        for s in range(len(tm)):

            for e in range(len(tm[0])):
                if s != e and tm[s, e] != 0:

                    edge_tuple = (s, e)
                    matrix_prob = tm[s, e]

                    if round(matrix_prob, 2) != matrix_prob:
                        matrix_prob = round(matrix_prob, 2)

                    v_c = self.vertices[edge_tuple[0]].get_center()
                    u_c = self.vertices[edge_tuple[1]].get_center()
                    vec = v_c - u_c
                    unit_vec = vec / np.linalg.norm(vec)

                    arrow_start = v_c - unit_vec * (
                        self.vertices[edge_tuple[0]].width / 2
                    )

                    label = (
                        Text(str(matrix_prob), font=REDUCIBLE_MONO)
                        .set_stroke(BLACK, width=8, background=True, opacity=0.8)
                        .scale(0.3)
                        .move_to(self.edges[edge_tuple])
                        .move_to(
                            arrow_start,
                            coor_mask=[0.8, 0.8, 0.8],
                        )
                    )

                    def label_updater(label):
                        label.move_to(self.edges[edge_tuple]).move_to(
                            self.vertices[edge_tuple[0]],
                            coor_mask=[0.6, 0.6, 0.6],
                        )

                    labels.add(label)
                    self.labels.append((label, edge_tuple))

        def update_labels(graph):
            for l, e in graph.labels:
                v_c = self.vertices[e[0]].get_center()
                u_c = self.vertices[e[1]].get_center()
                vec = v_c - u_c
                unit_vec = vec / np.linalg.norm(vec)

                arrow_start = v_c - unit_vec * (self.vertices[e[0]].width / 2)
                l.move_to(graph.edges[e]).move_to(
                    arrow_start,
                    coor_mask=[0.8, 0.8, 0.8],
                )

        self.add_updater(update_labels)

        return labels


class MarkovChainSimulator(Mobject):
    def __init__(
        self, markov_chain: MarkovChain, markov_chain_g: MarkovChainGraph, num_users=50
    ):
        super().__init__()
        self.set_opacity(0)
        self.markov_chain = markov_chain
        self.markov_chain_g = markov_chain_g
        self.num_users = num_users
        self.state_counts = {i: 0 for i in markov_chain.get_states()}
        self.init_users()

    def init_users(self):
        self.user_to_state = {
            i: np.random.choice(
                self.markov_chain.get_states(), p=self.markov_chain.get_current_dist()
            )
            for i in range(self.num_users)
        }
        for user_id in self.user_to_state:
            self.state_counts[self.user_to_state[user_id]] += 1

        self.users = [
            Dot(radius=0.035)
            .set_color(REDUCIBLE_YELLOW)
            .set_opacity(0.6)
            .set_stroke(REDUCIBLE_YELLOW, width=2, opacity=0.8)
            for _ in range(self.num_users)
        ]

        for user_id, user in enumerate(self.users):
            user_location = self.get_user_location(user_id)
            user.move_to(user_location)

        def user_updater(simulator: MarkovChainSimulator):
            for uid, u in enumerate(simulator.users):
                user_location = self.get_user_location(uid)
                u.move_to(user_location)

        self.add_updater(user_updater)

    def get_user_location(self, user: int):
        user_state = self.user_to_state[user]
        user_location = self.markov_chain_g.vertices[user_state].get_center()
        distributed_point = self.poisson_distribution(
            user, self.markov_chain_g.vertices[user_state]
        )

        user_location = [distributed_point[0], distributed_point[1], 0.0]

        return user_location

    def get_users(self):
        return self.users

    def transition(self):
        for user_id in self.user_to_state:
            self.user_to_state[user_id] = self.update_state(user_id)

    def update_state(self, user_id: int):
        current_state = self.user_to_state[user_id]
        transition_matrix = self.markov_chain.get_transition_matrix()
        new_state = np.random.choice(
            self.markov_chain.get_states(), p=transition_matrix[current_state]
        )
        self.state_counts[new_state] += 1
        return new_state

    def get_state_counts(self):
        return self.state_counts

    def get_user_dist(self, round_val=False):
        dist = {}
        total_counts = sum(self.state_counts.values())
        for user_id, count in self.state_counts.items():
            dist[user_id] = self.state_counts[user_id] / total_counts
            if round_val:
                dist[user_id] = round(dist[user_id], 2)
        return dist

    def get_instant_transition_animations(self):
        transition_animations = []
        self.transition()
        for user_id, user in enumerate(self.users):
            new_location = self.get_user_location(user_id)
            transition_animations.append(user.animate.move_to(new_location))
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

    def poisson_distribution(self, uid, state: VMobject):
        """
        This function creates a poisson distribution that places
        users around the center of the given state,
        particularly across the state's stroke.
        Implementation taken from: https://github.com/hpaulkeeler/posts/blob/master/PoissonCircle/PoissonCircle.py
        """

        np.random.seed(uid)

        radius = state.width / 2

        xxRand = np.random.normal(0, 1, size=(1, 2))

        # generate two sets of normal variables
        normRand = np.linalg.norm(xxRand, 2, 1)

        # Euclidean norms
        xxRandBall = xxRand / normRand[:, None]

        # rescale by Euclidean norms
        xxRandBall = radius * xxRandBall

        # rescale for non-unit sphere
        # retrieve x and y coordinates
        xx = xxRandBall[:, 0]
        yy = xxRandBall[:, 1]

        # Shift centre of circle to (xx0,yy0)
        center = state.get_center()
        xx = xx + center[0]
        yy = yy + center[1]

        return (xx[0], yy[0])


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

        markov_chain_g = MarkovChainGraph(
            markov_chain, enable_curved_double_arrows=False
        )
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        self.play(FadeIn(markov_chain_g), FadeIn(markov_chain_t_labels))
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=50
        )
        users = markov_chain_sim.get_users()

        self.play(*[FadeIn(user) for user in users])
        self.wait()

        num_steps = 10
        for _ in range(num_steps):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            self.play(*transition_animations)
        self.wait()

        for _ in range(num_steps):
            transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )
            self.wait()


class MarkovChainIntro(Scene):
    def construct(self):
        web_markov_chain, web_graph = self.get_web_graph()
        self.add(web_graph)
        self.wait()

        # self.start_simulation(web_markov_chain, web_graph)

    def get_web_graph(self):
        graph_layout = self.get_web_graph_layout()
        graph_edges = self.get_web_graph_edges(graph_layout)

        print(len(graph_layout))
        markov_chain = MarkovChain(len(graph_layout), graph_edges)
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            enable_curved_double_arrows=False,
            labels=False,
            layout=graph_layout,
        )

        return markov_chain, markov_chain_g

    def get_web_graph_layout(self):
        grid_height = 8
        grid_width = 12

        layout = {}
        node_id = 0
        STEP = 0.5
        for i in np.arange(-grid_height // 2, grid_height // 2, STEP):
            for j in np.arange(-grid_width // 2, grid_width // 2, STEP):
                noise = RIGHT * np.random.uniform(-1, 1) + UP * np.random.uniform(-1, 1)
                layout[node_id] = UP * i + RIGHT * j + noise * STEP / 3.1
                node_id += 1

        return layout

    def get_web_graph_edges(self, graph_layout):
        edges = []
        for u in graph_layout:
            for v in graph_layout:
                if u != v and np.linalg.norm(graph_layout[v] - graph_layout[u]) < 0.9:
                    if np.random.uniform() < 0.7:
                        edges.append((u, v))
        return edges

    def start_simulation(self, markov_chain, markov_chain_g):
        markov_chain_sim = MarkovChainSimulator(
            markov_chain,
            markov_chain_g,
            num_users=5000,
            user_radius=0.01,
        )
        users = markov_chain_sim.get_users()

        self.add(*users)
        self.wait()

        num_steps = 10
        # for _ in range(num_steps):
        #     transition_animations = markov_chain_sim.get_instant_transition_animations()
        #     self.play(*transition_animations)
        # self.wait()

        for _ in range(num_steps):
            transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )


class IntroImportanceProblem(Scene):
    def construct(self):
        pass


class IntroStationaryDistribution(Scene):
    def construct(self):
        self.show_counts()

    def show_counts(self):
        markov_chain = MarkovChain(
            5,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
                (2, 4),
                (1, 4),
                (4, 2),
                (3, 4),
                (4, 0),
            ],
        )
        markov_chain_g = MarkovChainGraph(
            markov_chain, enable_curved_double_arrows=False, layout="circular"
        )
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        # markov_chain_g.scale(1.5)
        self.play(
            FadeIn(markov_chain_g),
            # FadeIn(markov_chain_t_labels)
        )
        self.wait()
        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=1
        )
        users = markov_chain_sim.get_users()
        # scale user a bit here
        users[0].scale(1.5)

        self.play(*[FadeIn(user) for user in users])
        self.wait()

        num_steps = 10
        print("Count", markov_chain_sim.get_state_counts())
        print("Dist", markov_chain_sim.get_user_dist())
        count_labels = self.get_current_count_mobs(markov_chain_g, markov_chain_sim)
        self.play(*[FadeIn(label) for label in count_labels.values()])
        self.wait()
        use_dist = False
        for i in range(num_steps):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            self.play(*transition_animations + count_transforms)
            if i < 5:
                self.wait()
            if i > 20:
                use_dist = True
            print("Iteration", i)
            print("Count", markov_chain_sim.get_state_counts())
            print("Dist", markov_chain_sim.get_user_dist())

        true_stationary_dist = markov_chain.get_true_stationary_dist()
        print("True stationary dist", true_stationary_dist)
        print("Norm:", np.linalg.norm(true_stationary_dist))

    def get_current_count_mobs(self, markov_chain_g, markov_chain_sim, use_dist=False):
        vertex_mobs_map = markov_chain_g.vertices
        count_labels = {}
        for v in vertex_mobs_map:
            if not use_dist:
                state_counts = markov_chain_sim.get_state_counts()
                label = Text(str(state_counts[v]), font="SF Mono").scale(0.6)
            else:
                state_counts = markov_chain_sim.get_user_dist(round_val=True)
                label = Text("{0:.2f}".format(state_counts[v]), font="SF Mono").scale(
                    0.6
                )
            label_direction = normalize(
                vertex_mobs_map[v].get_center() - markov_chain_g.get_center()
            )
            label.next_to(vertex_mobs_map[v], label_direction)
            count_labels[v] = label

        return count_labels

    def update_count_labels(
        self, count_labels, markov_chain_g, markov_chain_sim, use_dist=False
    ):
        if count_labels is None:
            count_labels = self.get_current_count_mobs(
                markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            transforms = [Write(label) for label in count_labels.values()]

        else:
            new_count_labels = self.get_current_count_mobs(
                markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            transforms = [
                Transform(count_labels[v], new_count_labels[v]) for v in count_labels
            ]

        return count_labels, transforms


class StationaryDistPreview(Scene):
    def construct(self):
        stationary_dist = Text(
            "Stationary Distribution", font="CMU Serif", weight=BOLD
        ).scale(0.8)
        point_1 = Text(
            "1. How to find stationary distributions?", font="CMU Serif"
        ).scale(0.5)
        point_2 = Text("2. When do they exist?", font="CMU Serif").scale(0.5)
        point_3 = Text("3. How do we efficiently compute them?").scale(0.5)
        points = VGroup(point_1, point_2, point_3).arrange(DOWN, aligned_edge=LEFT)

        text = VGroup(stationary_dist, points).arrange(DOWN)

        text.move_to(LEFT * 3.5)

        self.play(Write(text[0]))
        self.wait()

        self.play(FadeIn(point_1))
        self.wait()

        self.play(FadeIn(point_2))
        self.wait()

        self.play(FadeIn(point_3))
        self.wait()
