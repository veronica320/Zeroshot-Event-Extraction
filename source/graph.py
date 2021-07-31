## A graph of events. Adapted from OneIE. ##

class Graph(object):
	def __init__(self, triggers, roles, vocabs):
		"""
		:param triggers (list): A list of triggers represented as a tuple of
		(start_offset, end_offset, label_idx). end_offset = the index of the end
		token + 1.
		:param roles: A list of roles represented as a tuple of (trigger_idx_1,
		start_offset, end_offset, label_idx).
		:param vocabs (dict): Label type vocabularies.
		"""
		self.triggers = triggers
		self.roles = roles
		self.vocabs = vocabs

		self.trigger_num = len(triggers)
		self.role_num = len(roles)
		self.graph_local_score = 0.0

		# subscores
		self.trigger_scores = []
		self.role_scores = []

	def __eq__(self, other):
		if isinstance(other, Graph):
			equal = (self.triggers == other.triggers and
			         self.roles == other.roles)
			return equal
		return False

	def to_dict(self):
		"""Convert a graph to a dict object
		:return (dict): A dictionary representing the graph, where label indices
		have been replaced with label strings.
		"""
		trigger_itos = {i: s for s, i in self.vocabs['event_type'].items()}
		role_itos = {i: s for s, i in self.vocabs['role_type'].items()}

		triggers = [[i, j, trigger_itos[k], l] for (i, j, k), l in zip(self.triggers, self.trigger_scores)]
		roles = [[h, i, j, role_itos[k], l] for (h, i, j, k), l in zip(self.roles, self.role_scores)]

		return {
			'triggers': triggers,
			'roles': roles,
		}

	def __str__(self):
		return str(self.to_dict())

	def copy(self):
		"""Make a copy of the graph
		:return (Graph): a copy of the current graph.
		"""
		graph = Graph(
			triggers=self.triggers.copy(),
			roles=self.roles.copy(),
			vocabs=self.vocabs
		)
		graph.graph_local_score = self.graph_local_score
		graph.trigger_scores = self.trigger_scores
		graph.role_scores = self.role_scores
		return graph

	def add_trigger(self, start, end, label, score=0, score_norm=0):
		"""Add an event trigger to the graph.
		:param start (int): Start token offset of the trigger.
		:param end (int): End token offset of the trigger + 1.
		:param label (int): Index of the event type label.
		:param score (float): Label score.
		"""
		self.triggers.append((start, end, label))
		self.trigger_num = len(self.triggers)
		self.graph_local_score += score
		self.trigger_scores.append(score_norm)

	def add_role(self, idx1, start, end, label, score=0, score_norm=0):
		"""Add an event-argument link edge to the graph.
		:param idx1 (int): Index of the trigger node.
		:param start (int): Start token offset.
		:param end (int): End token offset + 1.
		:param label (int): Index of the role label.
		:param score (float): Label score.
		"""
		if label:
			self.roles.append((idx1, start, end, label))
			self.role_scores.append(score_norm)
		self.role_num = len(self.roles)
		self.graph_local_score += score

	@staticmethod
	def empty_graph(vocabs):
		"""Create a graph without any node and edge.
		:param vocabs (dict): Vocabulary object.
		"""
		return Graph([], [], [], [], vocabs)

