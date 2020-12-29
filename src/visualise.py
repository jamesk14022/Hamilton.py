import numpy as np
import attributes as att
from graphics import *

class Visualise:
	buffer = 1.05
	window_height, window_width = 800, 800
	np.set_printoptions(precision=3)


	def render_graph(pos, g):
		win = GraphWin(width = Visualise.window_width, height = Visualise.window_width) # create a window
		# coords on x and y axis are between 0 and 100 
		win.setCoords(np.amin(pos) * 1.1, np.amin(pos) * 1.1, np.amax(pos) * 1.1, np.amax(pos) * 1.1) 

		circle_size = (np.amax(pos) * 1.1 + abs(np.amin(pos) * 1.1)) / 150 
		# then render edges 
		ui = np.triu_indices(g.nodes, k = 1)
		for e in range(0, len(ui[0])):
			if g.adj[ui[0][e], ui[1][e]] == 1:
				l = Line(Point(pos[ui[0][e],0], pos[ui[0][e], 1]), Point(pos[ui[1][e], 0], pos[ui[1][e], 1]))
				l.draw(win)
		# render nodes
		for i in range(0, g.nodes):
			c = Circle(Point(pos[i, 0], pos[i, 1]), circle_size)	
			c.draw(win) 
		win.getMouse() # pause before closing

	# draw a graph using the spectral layout
	# https://en.wikipedia.org/wiki/Spectral_layout
	def draw_spectral(g):
		L = att.get_laplace(g)
		eigenvalue, eigenvector = np.linalg.eig(L)

		# returns the indices of the two smallest eigenvalues
		minind = np.argsort(eigenvalue)[1 : 3]
		xeig = eigenvector[:,minind[0]]
		yeig = eigenvector[:,minind[1]]

		pos = np.column_stack((xeig, yeig))
		Visualise.render_graph(pos, g)

	# what does the threshold parameter here do?
	def draw_fruchterman_reingold(g, iterations=50, threshold=1e-4, dim=2):
		A = g.adj
		nnodes, _ = A.shape

		# random initial positions
		pos = np.asarray(np.random.rand(nnodes, dim), dtype=A.dtype)
		# optimal distance between nodes
		k = np.sqrt(1.0 / nnodes)
		# this is the largest step allowed in the dynamics.
		t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
		# discretization of time
		dt = t / float(iterations + 1)
		delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
		for iteration in range(iterations):
			# matrix of difference between points
			delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
			# distance between points
			distance = np.linalg.norm(delta, axis=-1)
			# enforce minimum distance of 0.01
			np.clip(distance, 0.01, None, out=distance)
			# displacement "force"
			displacement = np.einsum(
				"ijk,ij->ik", delta, (k * k / distance ** 2 - A * distance / k)
			)
			# update positions
			length = np.linalg.norm(displacement, axis=-1)
			length = np.where(length < 0.01, 0.1, length)
			# I don't quite understand this line, research
			delta_pos = np.einsum("ij,i->ij", displacement, t / length)
			pos += delta_pos
			t -= dt
			err = np.linalg.norm(delta_pos) / nnodes
			if err < threshold:
				break

		Visualise.render_graph(pos, g)


