
   
import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from reducible_colors import *
from functions import *

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

        # handle sink nodes to point to itself
        for i, row in enumerate(self.transition_matrix):
            if np.sum(row) == 0:
                self.transition_matrix[i][i] = 1

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
        dist = np.linalg.eig(np.transpose(self.transition_matrix))[1][:,0]
        return dist / sum(dist)

class CustomLabel(Text):
    def __init__(self, label, font="SF Mono", scale=1, weight=BOLD):
        super().__init__(label, font=font, weight=weight)
        self.scale(scale)


class CustomCurvedArrow(CurvedArrow):
    def __init__(self, start, end, tip_length=0.15, **kwargs):
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
            labels={
                k: CustomLabel(str(k), scale=0.6) for k in markov_chain.get_states()
            }
        

        self.labels = {}

        super().__init__(
            markov_chain.get_states(),
            markov_chain.get_edges(),
            vertex_config=vertex_config,
            labels=labels,
            **kwargs
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
                
                u_radius = self.vertices[u].width / 2
                v_radius = self.vertices[v].width / 2

                arrow_start = u_c + unit_vec * u_radius
                arrow_end = v_c - unit_vec * v_radius
                edge.put_start_and_end_on(arrow_start, arrow_end)

        self.add_updater(update_edges)
        update_edges(self)

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
            arrow_start = u_c + unit_vec * self.vertices[u].radius
            arrow_end = v_c - unit_vec * self.vertices[v].radius
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

        print(straight_edge_config)

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

                    label = (
                        Text(str(matrix_prob), font=REDUCIBLE_MONO)
                        .set_stroke(BLACK, width=8, background=True, opacity=0.8)
                        .scale(0.3)
                        .move_to(self.edges[edge_tuple].point_from_proportion(0.2))
                        )

                    labels.add(label)
                    self.labels[edge_tuple] = label

        def update_labels(graph):
            for e, l in graph.labels.items():
                l.move_to(graph.edges[e].point_from_proportion(0.2))

        self.add_updater(update_labels)

        return labels


class MarkovChainSimulator:
    def __init__(
        self, markov_chain: MarkovChain, markov_chain_g: MarkovChainGraph, num_users=50, user_radius=0.035,
    ):
        self.markov_chain = markov_chain
        self.markov_chain_g = markov_chain_g
        self.num_users = num_users
        self.state_counts = {i: 0 for i in markov_chain.get_states()}
        self.user_radius = user_radius
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
            Dot(radius=self.user_radius)
            .set_color(REDUCIBLE_YELLOW)
            .set_opacity(0.6)
            .set_stroke(REDUCIBLE_YELLOW, width=2, opacity=0.8)
            for _ in range(self.num_users)
        ]

        for user_id, user in enumerate(self.users):
            user_location = self.get_user_location(user_id)
            user.move_to(user_location)

    def get_user_location(self, user: int):
        user_state = self.user_to_state[user]
        user_location = self.markov_chain_g.vertices[user_state].get_center()
        distributed_point = self.poisson_distribution(user_location)

        user_location = [distributed_point[0], distributed_point[1], 0.0]

        return user_location

    def get_users(self):
        return self.users

    def transition(self):
        for user_id in self.user_to_state:
            self.user_to_state[user_id] = self.update_state(user_id)
        self.markov_chain.update_dist()

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

    def poisson_distribution(self, center):
        """
        This function creates a poisson distribution that places
        users around the center of the given state,
        particularly across the state's stroke.
        Implementation taken from: https://github.com/hpaulkeeler/posts/blob/master/PoissonCircle/PoissonCircle.py
        """

        radius = self.markov_chain_g.vertices[0].width / 2

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

        markov_chain_g = MarkovChainGraph(markov_chain, enable_curved_double_arrows=True)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        self.play(
            FadeIn(markov_chain_g),
            FadeIn(markov_chain_t_labels)
        )
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

### BEGIN INTRODUCTION.mp4 ###
class IntroWebGraph(Scene):
    def construct(self):
        web_markov_chain, web_graph = self.get_web_graph()
        self.add(web_graph)
        self.wait()

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

class UserSimulationWebGraph(IntroWebGraph):
    def construct(self):
        web_markov_chain, web_graph = self.get_web_graph()
        self.start_simulation(web_markov_chain, web_graph)

    def start_simulation(self, markov_chain, markov_chain_g):
        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=2000, user_radius=0.01,
        )
        users = markov_chain_sim.get_users()

        self.add(*users)
        self.wait()

        num_steps = 10

        for _ in range(num_steps):
            transforms = markov_chain_sim.get_instant_transition_animations()
            self.play(
                *transforms, rate_func=linear,
            )

        # for _ in range(num_steps):
        #     transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
        #     self.play(
        #         *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
        #     )

class MarkovChainPageRankTitleCard(Scene):
    def construct(self):
        title = Text("Markov Chains", font="CMU Serif", weight=BOLD).move_to(UP * 3.5)
        self.play(
            Write(title)
        )
        self.wait()

        pagerank_title = Text("PageRank", font="CMU Serif", weight=BOLD).move_to(UP * 3.5)

        self.play(
            ReplacementTransform(title, pagerank_title)
        )
        self.wait()

### END INTRODUCTION.mp4 ###

class MarkovChainIntro(Scene):
    def construct(self):
        markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1), (3, 2)],
        )

        markov_chain_g = MarkovChainGraph(markov_chain, enable_curved_double_arrows=True)
        markov_chain_g.scale(1.2)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        self.play(
            FadeIn(markov_chain_g),
            FadeIn(markov_chain_t_labels)
        )
        self.wait()

        self.highlight_states(markov_chain_g)

        transition_probs = self.highlight_transitions(markov_chain_g)

        self.highlight_edge(markov_chain_g, (3, 1))
        p_3_1 = Tex(r"$P(3, 1)$ = 0.5").scale(0.8)
        p_2_3 = Tex(r"$P(2, 3)$ = 1.0").scale(0.8)
        VGroup(p_3_1, p_2_3).arrange(DOWN).move_to(LEFT * 4)

        self.play(
            FadeIn(p_3_1)
        )
        self.wait()

        self.highlight_edge(markov_chain_g, (2, 3))

        self.play(
            FadeIn(p_2_3)
        )
        self.wait()

        reset_animations = self.reset_edges(markov_chain_g)
        self.play(
            *reset_animations,
            FadeOut(p_3_1),
            FadeOut(p_2_3),
            FadeOut(transition_probs)
        )
        self.wait()

        self.discuss_markov_prop(markov_chain_g)

    def highlight_states(self, markov_chain_g):
        highlight_animations = []
        for edge in markov_chain_g.edges.values():
            highlight_animations.append(
                edge.animate.set_stroke(opacity=0.5)
            )
            highlight_animations.append(
                edge.tip.animate.set_fill(opacity=0.5).set_stroke(opacity=0.5)
            )
        for label in markov_chain_g.labels.values():
            highlight_animations.append(
                label.animate.set_fill(opacity=0.5)
            )
        glowing_circles = []
        for vertex in markov_chain_g.vertices.values():
            glowing_circle = get_glowing_surround_circle(vertex)
            highlight_animations.append(
                FadeIn(glowing_circle)
            )
            glowing_circles.append(glowing_circle)

        states = Text("States", font="CMU Serif").move_to(UP * 3.5).set_color(REDUCIBLE_YELLOW)
        arrow_1 = Arrow(states.get_bottom(), markov_chain_g.vertices[2])
        arrow_2 = Arrow(states.get_bottom(), markov_chain_g.vertices[0])
        arrow_1.set_color(GRAY)
        arrow_2.set_color(GRAY)

        self.play(
            *highlight_animations,
        )
        self.wait()

        self.play(
            Write(states),
            Write(arrow_1),
            Write(arrow_2)
        )
        self.wait()

        un_highlight_animations = []
        for edge in markov_chain_g.edges.values():
            un_highlight_animations.append(
                edge.animate.set_stroke(opacity=1)
            )
            un_highlight_animations.append(
                edge.tip.animate.set_fill(opacity=1).set_stroke(opacity=1)
            )
        for label in markov_chain_g.labels.values():
            un_highlight_animations.append(
                label.animate.set_fill(opacity=1)
            )

        for v in markov_chain_g.vertices:
            un_highlight_animations.append(
                FadeOut(glowing_circles[v])
            )

        self.play(
            *un_highlight_animations,
            FadeOut(states),
            FadeOut(arrow_1),
            FadeOut(arrow_2)
        )
        self.wait()

    def highlight_transitions(self, markov_chain_g):
        self.play(
            *[label.animate.set_color(REDUCIBLE_YELLOW) for label in markov_chain_g.labels.values()]
        )
        self.wait()

        transition_probs = Tex("Transition Probabilities $P(i, j)$").set_color(REDUCIBLE_YELLOW)
        transition_probs.move_to(UP * 3.5)
        self.play(
            FadeIn(transition_probs)
        )
        self.wait()

        return transition_probs

    def highlight_edge(self, markov_chain_g, edge_tuple):
        highlight_animations = []
        for edge in markov_chain_g.edges:
            if edge == edge_tuple:
                highlight_animations.extend(
                    [
                    markov_chain_g.edges[edge].animate.set_stroke(opacity=1),
                    markov_chain_g.edges[edge].tip.animate.set_stroke(opacity=1).set_fill(opacity=1),
                    markov_chain_g.labels[edge].animate.set_fill(color=REDUCIBLE_YELLOW, opacity=1),
                    ]
                )
            else:
                highlight_animations.extend(
                    [
                    markov_chain_g.edges[edge].animate.set_stroke(opacity=0.3),
                    markov_chain_g.edges[edge].tip.animate.set_stroke(opacity=0.3).set_fill(opacity=0.3),
                    markov_chain_g.labels[edge].animate.set_fill(color=WHITE, opacity=0.3)
                    ]
                )
        self.play(
            *highlight_animations
        )

    def reset_edges(self, markov_chain_g):
        un_highlight_animations = []
        for edge in markov_chain_g.edges.values():
            un_highlight_animations.append(
                edge.animate.set_stroke(opacity=1)
            )
            un_highlight_animations.append(
                edge.tip.animate.set_fill(opacity=1).set_stroke(opacity=1)
            )
        for label in markov_chain_g.labels.values():
            un_highlight_animations.append(
                label.animate.set_fill(color=WHITE, opacity=1)
            )
        return un_highlight_animations

    def discuss_markov_prop(self, markov_chain_g):
        markov_prop_explained = Tex("Transition probability only depends \\\\ on current state and future state").scale(0.8)
        markov_prop_explained.move_to(UP * 3.5)

        self.play(
            FadeIn(markov_prop_explained)
        )
        self.wait()

        user_1 = Dot().set_color(REDUCIBLE_GREEN_DARKER).set_stroke(color=REDUCIBLE_GREEN_LIGHTER, width=2)
        user_2 = Dot().set_color(REDUCIBLE_YELLOW_DARKER).set_stroke(color=REDUCIBLE_YELLOW, width=2)

        user_1_label = user_1.copy()
        user_1_transition = MathTex(r"2 \rightarrow 3").scale(0.7)
        user_1_label_trans = VGroup(user_1_label, user_1_transition).arrange(RIGHT)
        user_2_label = user_2.copy()
        user_2_transition = MathTex(r"1 \rightarrow 3").scale(0.7)
        user_2_label_trans = VGroup(user_2_label, user_2_transition).arrange(RIGHT)

        result = Tex("For both users").scale(0.7)
        result_with_dots = VGroup(result, user_1_label.copy(), user_2_label.copy()).arrange(RIGHT)
        p_3_1 = Tex(r"$P(3, 1)$ = 0.5").scale(0.7)
        p_3_2 = Tex(r"$P(3, 2)$ = 0.5").scale(0.7)

        left_text = VGroup(user_1_label_trans, user_2_label_trans, result_with_dots, p_3_1, p_3_2).arrange(DOWN).to_edge(LEFT * 2)
        user_1.next_to(markov_chain_g.vertices[2], LEFT, buff=SMALL_BUFF)
        user_2.next_to(markov_chain_g.vertices[1], DOWN, buff=SMALL_BUFF)

        self.play(
            FadeIn(user_1),
        )
        self.wait()
        self.play(
            FadeIn(user_1_label_trans)
        )
        self.wait()

        self.play(
            user_1.animate.next_to(markov_chain_g.vertices[3], LEFT, buff=SMALL_BUFF)
        )
        self.wait()

        self.play(
            FadeIn(user_2),
            FadeIn(user_2_label_trans)
        )
        self.wait()

        self.play(
            user_2.animate.next_to(markov_chain_g.vertices[3], DOWN, buff=SMALL_BUFF)
        )
        self.wait()

        self.play(
            FadeIn(result_with_dots)
        )
        self.wait()
        highlight_animations = []

        for edge in markov_chain_g.edges:
            if edge == (3, 2) or edge == (3, 1):
                highlight_animations.extend(
                    [
                    markov_chain_g.labels[edge].animate.set_color(REDUCIBLE_YELLOW)
                    ]
                )
            else:
                highlight_animations.extend(
                    [
                    markov_chain_g.labels[edge].animate.set_fill(opacity=0.3),
                    markov_chain_g.edges[edge].animate.set_stroke(opacity=0.3),
                    markov_chain_g.edges[edge].tip.animate.set_fill(opacity=0.3).set_stroke(opacity=0.3)
                    ]
                )

        self.play(
            FadeIn(p_3_1),
            FadeIn(p_3_2),
            *highlight_animations
        )
        self.wait()

        markov_property = Text("Markov Property", font="CMU Serif", weight=BOLD).scale(0.8).move_to(DOWN * 3.5)

        self.play(
            Write(markov_property)
        )
        self.wait()


class IntroImportanceProblem(Scene):
    def construct(self):
        title = Text("Ranking States", font="CMU Serif", weight=BOLD)
        title.move_to(UP * 3.5)

        self.play(
            Write(title)
        )
        self.wait()

        markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1), (3, 2)],
        )

        markov_chain_g = MarkovChainGraph(markov_chain, enable_curved_double_arrows=True, layout="circular")
        markov_chain_g.scale(1.1)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        self.play(
            FadeIn(markov_chain_g)   
        )
        self.wait()

        base_ranking_values = [0.95, 0.75, 0.5, 0.25]
        original_width = markov_chain_g.vertices[0].width
        final_ranking = self.show_randomized_ranking(markov_chain_g, base_ranking_values)

        

        how_to_measure_importance = Text("How to Measure Relative Importance?", font="CMU Serif", weight=BOLD).scale(0.8)
        how_to_measure_importance.move_to(title.get_center())
        self.play(
            *[
            markov_chain_g.vertices[v].animate.scale_to_fit_width(original_width) for v in markov_chain.get_states()
            ],
            FadeOut(final_ranking),
            ReplacementTransform(title, how_to_measure_importance),
        )
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=100
        )
        users = markov_chain_sim.get_users()

        self.play(*[FadeIn(user) for user in users])
        self.wait()

        num_steps = 5
        for _ in range(num_steps):
            transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )
            self.wait()
        self.wait()

    def show_randomized_ranking(self, markov_chain_g, base_ranking_values):
        original_markov_chain_nodes = [markov_chain_g.vertices[i].copy() for i in range(len(base_ranking_values))]
        positions = [LEFT * 2.4, LEFT * 0.8, RIGHT * 0.8, RIGHT * 2.4]
        gt_signs = [MathTex(">"), MathTex(">"), MathTex(">")]
        for i, sign in enumerate(gt_signs):
            gt_signs[i].move_to((positions[i] + positions[i + 1]) / 2)
        num_iterations = 5
        SHIFT_DOWN = DOWN * 3.2
        for step in range(num_iterations):
            print('Iteration', step)
            current_ranking_values = self.generate_new_ranking(base_ranking_values)
            current_ranking_map = self.get_ranking_map(current_ranking_values)
            scaling_animations = []
            for v, scaling in current_ranking_map.items():
                scaling_animations.append(
                    markov_chain_g.vertices[v].animate.scale_to_fit_width(scaling)
                )
            current_ranking = self.get_ranking(current_ranking_map)
            ranking_animations = []
            for i, v in enumerate(current_ranking):
                if step != 0:
                    ranking_animations.append(
                        original_markov_chain_nodes[v].animate.move_to(positions[i] + SHIFT_DOWN)
                    )
                else:
                    ranking_animations.append(
                        FadeIn(original_markov_chain_nodes[v].move_to(positions[i] + SHIFT_DOWN))
                    )
            
            if step == 0:
                ranking_animations.extend(
                    [FadeIn(sign.shift(SHIFT_DOWN)) for sign in gt_signs]
                )

            self.play(
                *scaling_animations + ranking_animations
            )
            self.wait()
            
        return VGroup(*original_markov_chain_nodes + gt_signs)



    def get_ranking(self, ranking_map):
        sorted_map = {k: v for k, v in sorted(ranking_map.items(), key=lambda item: item[1])}
        return [key for key in sorted_map][::-1]


    def generate_new_ranking(self, ranking_values):
        np.random.shuffle(ranking_values)
        new_ranking = []
        for elem in ranking_values:
            new_ranking.append(elem + np.random.uniform(-0.08, 0.08))
        return new_ranking

    def get_ranking_map(self, ranking_values):
        return {i: ranking_values[i] for i in range(len(ranking_values))}


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
        markov_chain_g = MarkovChainGraph(markov_chain, enable_curved_double_arrows=True, layout="circular")
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        markov_chain_g.scale(1.5)
        self.play(
            FadeIn(markov_chain_g),
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

        num_steps = 50
        stabilize_threshold = num_steps - 20
        print('Count', markov_chain_sim.get_state_counts())
        print('Dist', markov_chain_sim.get_user_dist())
        count_labels = self.get_current_count_mobs(markov_chain_g, markov_chain_sim)
        self.play(*[FadeIn(label) for label in count_labels.values()])
        self.wait()
        use_dist = False
        for i in range(num_steps):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            if i > stabilize_threshold:
                self.play(
                    *transition_animations
                )
                continue
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
