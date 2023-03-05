

class DisjointSet:
	"""Data structure that makes it easy to figure out if somebody
	won in a game of Hex"""

	def __init__(self, elements):
		self.elements = elements
		self.parent = {}
		for element in elements:
			self.make_set(element)


	def make_set(self, a):
		self.parent[a] = a


	def find(self, a):
		if self.parent[a] == a:
			return a
		else:
			"""Using path compression for later lookups"""
			self.parent[a] = self.find(self.parent[a])
			return self.parent[a]


	def join(self, a, b):
		root_a = self.find(a)
		root_b = self.find(b)
		if root_a == root_b:
			return
		else:
			self.parent[root_b] = root_a
		# Complexity can be improved here...