from manimlib.imports import *

from math import sin,cos

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flat -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)

class makeText(Scene):
	def construct(self):
		#######Code#######
		#Making text
		first_line = TextMobject("Manim is fun")
		second_line = TextMobject("and useful")
		final_line = TextMobject("Hope you like it too!", color=BLUE)
		color_final_line = TextMobject("Hope you like it too!")

		#Coloring
		color_final_line.set_color_by_gradient(BLUE,PURPLE)

		#Position text
		second_line.next_to(first_line, DOWN)

		#Showing text
		self.wait(3)
		self.play(Write(first_line), Write(second_line))
		self.wait(1)
		self.play(FadeOut(second_line), ReplacementTransform(first_line, final_line))
		self.wait(1)
		self.play(Transform(final_line, color_final_line))
		self.wait(2)


class Equations(Scene):
	def construct(self):
		#Making equations
		first_eq = TextMobject("$$J(\\theta) = -\\frac{1}{m} [\\sum_{i=1}^{m} y^{(i)} \\log{h_{\\theta}(x^{(i)})} + (1-y^{(i)}) \\log{(1-h_{\\theta}(x^{(i)}))}] $$")
		second_eq = ["$J(\\theta_{0}, \\theta_{1})$", "=", "$\\frac{1}{2m}$", "$\\sum\\limits_{i=1}^m$", "(", "$h_{\\theta}(x^{(i)})$", "-", "$y^{(i)}$", "$)^2$"]

		second_mob = TextMobject(*second_eq)

		for i,item in enumerate(second_mob):
			if(i != 0):
				item.next_to(second_mob[i-1],RIGHT)

		eq2 = VGroup(*second_mob)

		des1 = TextMobject("With manim, you can write complex equations like this...")
		des2 = TextMobject("Or this...")
		des3 = TextMobject("And it looks nice!!")

		#Coloring equations
		second_mob.set_color_by_gradient("#33ccff","#ff00ff")

		#Positioning equations
		des1.shift(2*UP)
		des2.shift(2*UP)

		#Animating equations
		self.play(Write(des1))
		self.play(Write(first_eq))
		self.play(ReplacementTransform(des1, des2), Transform(first_eq, eq2))
		self.wait(1)

		for i, item in enumerate(eq2):
			if (i<2):
				eq2[i].set_color(color=PURPLE)
			else:
				eq2[i].set_color(color="#00FFFF")

		self.add(eq2)
		self.wait(1)
		self.play(FadeOutAndShiftDown(eq2), FadeOutAndShiftDown(first_eq), Transform(des2, des3))
		self.wait(2)


import math

class Graphing(GraphScene):
	CONFIG = {
		"x_min": -5,
		"x_max": 5,
		"y_min": -4,
		"y_max": 4,
		"graph_origin": ORIGIN,
		"function_color": WHITE,
		"axes_color": BLUE
	}

	def construct(self):
		#Make graph
		self.setup_axes(animate=True)
		func_graph=self.get_graph(self.func_to_graph,self.function_color)
		graph_lab = self.get_graph_label(func_graph, label = "x^{2}")

		func_graph_2=self.get_graph(self.func_to_graph_2,self.function_color)
		graph_lab_2 = self.get_graph_label(func_graph_2, label = "x^{3}")

		vert_line = self.get_vertical_line_to_graph(1,func_graph,color=YELLOW)

		x = self.coords_to_point(1, self.func_to_graph(1))
		y = self.coords_to_point(0, self.func_to_graph(1))
		horz_line = Line(x,y, color=YELLOW)

		point = Dot(self.coords_to_point(1,self.func_to_graph(1)))

		#Display graph
		self.play(ShowCreation(func_graph), Write(graph_lab))
		self.wait(1)
		self.play(ShowCreation(vert_line))
		self.play(ShowCreation(horz_line))
		self.add(point)
		self.wait(1)
		self.play(Transform(func_graph, func_graph_2), Transform(graph_lab, graph_lab_2))
		self.wait(2)


	def func_to_graph(self, x):
		return (x**2)

	def func_to_graph_2(self, x):
		return(x**3)

class ThreeDSurface(ParametricSurface):
	def __init__(self, **kwargs):
		kwargs = {
			"u_min": -2,
			"u_max": 2,
			"v_min": -2,
			"v_max": 2,
			"checkerboard_colors": [BLUE_D]
		}
		ParametricSurface.__init__(self, self.func, **kwargs)

	def func(self, x, y):
		return np.array([x,y,x**2 - y**2])


class Test(ThreeDScene):
	def construct(self):
		self.set_camera_orientation(0.6, -0.7853981, 86.6)

		surface = ThreeDSurface()
		self.play(ShowCreation(surface))

		d = Dot(np.array([0,0,0]), color = YELLOW)
		self.play(ShowCreation(d))


		self.wait()
		self.move_camera(0.8*np.pi/2, -0.45*np.pi)
		self.begin_ambient_camera_rotation()
		self.wait(9)




class SimpleField(Scene):
	CONFIG = {
	"plane_kwargs" : {
		"color" : RED
		},
	}
	def construct(self):
		plane = NumberPlane(**self.plane_kwargs)
		plane.add(plane.get_axis_labels())
		self.add(plane)

		points = [x*RIGHT+y*UP
			for x in np.arange(-5,5,1)
			for y in np.arange(-5,5,1)
			]
		
		vec_field = []
		for point in points:
			field = 0.5*RIGHT + 0.5*UP
			result = Vector(field).shift(point)
			vec_field.append(result)

		draw_field = VGroup(*vec_field)


		self.play(ShowCreation(draw_field))


class FieldOfMovingCharge(Scene):
	CONFIG = {
	"plane_kwargs" : {
		"color" : RED_B
		},
	"point_charge_start_loc" : 5.5*LEFT-1.5*UP,
	}
	def construct(self):
		plane = NumberPlane(**self.plane_kwargs)
		#plane.main_lines.fade(.9)
		plane.add(plane.get_axis_labels())
		self.add(plane)

		field = VGroup(*[self.create_vect_field(self.point_charge_start_loc,x*RIGHT+y*UP)
			for x in np.arange(-9,9,1)
			for y in np.arange(-5,5,1)
			])
		self.field=field
		self.source_charge = self.Positron().move_to(self.point_charge_start_loc)
		self.source_charge.velocity = np.array((1,0,0))
		self.play(FadeIn(self.source_charge))
		self.play(ShowCreation(field))
		self.moving_charge()

	def create_vect_field(self,source_charge,observation_point):
		return Vector(self.calc_field(source_charge,observation_point)).shift(observation_point)

	def calc_field(self,source_point,observation_point):
		x,y,z = observation_point
		Rx,Ry,Rz = source_point
		r = math.sqrt((x-Rx)**2 + (y-Ry)**2 + (z-Rz)**2)
		if r<0.0000001:   #Prevent divide by zero  ##Note:  This won't work - fix this
			efield = np.array((0,0,0))  
		else:
			efield = (observation_point - source_point)/r**3
		return efield



	def moving_charge(self):
		numb_charges=3
		possible_points = [v.get_start() for v in self.field]
		points = random.sample(possible_points, numb_charges)
		particles = VGroup(self.source_charge, *[
			self.Positron().move_to(point)
			for point in points
		])
		for particle in particles[1:]:
			particle.velocity = np.array((0,0,0))
		self.play(FadeIn(particles[1:]))
		self.moving_particles = particles
		self.add_foreground_mobjects(self.moving_particles )
		self.always_continually_update = True
		self.wait(10)


	def continual_update(self, *args, **kwargs):
		Scene.continual_update(self, *args, **kwargs)
		if hasattr(self, "moving_particles"):
			dt = self.frame_duration

			for v in self.field:
				field_vect=np.zeros(3)
				for p in self.moving_particles:
					field_vect = field_vect + self.calc_field(p.get_center(), v.get_start())
				v.put_start_and_end_on(v.get_start(), field_vect+v.get_start())

			for p in self.moving_particles:
				accel = np.zeros(3)
				p.velocity = p.velocity + accel*dt
				p.shift(p.velocity*dt)


	class Positron(Circle):
		CONFIG = {
		"radius" : 0.2,
		"stroke_width" : 3,
		"color" : RED,
		"fill_color" : RED,
		"fill_opacity" : 0.5,
		}
		def __init__(self, **kwargs):
			Circle.__init__(self, **kwargs)
			plus = TexMobject("+")
			plus.scale(0.7)
			plus.move_to(self)
			self.add(plus)

class Formula2(GraphScene):
	CONFIG = {
		"x_min" : 0,
		"x_max" : 10.3,
		"x_tick_frequency": 1,
		"y_min" : 0,
		"y_max" : 10.3,
		"y_tick_frequency": 1,
		"graph_origin" : [-4,-3,0] ,
		"function_color" : RED ,
		"axes_color" : WHITE ,
		"x_labeled_nums" : range(1,10,1),
		"y_labeled_nums" : range(1,10,1)
	}
	def construct(self):
		self.setup_axes(animate=False)
		func_graph = self.get_graph(self.func_to_graph, self.function_color)
		vert_line = Line(start=self.coords_to_point(1,0), color=YELLOW)
		vert_line.add_updater(
			lambda mob: mob.put_start_and_end_on(
					vert_line.get_start(),
					func_graph.get_end()
				)
			)


		self.play(ShowCreation(func_graph))
		# add the vert_line because it have a updater method
		self.add(vert_line)
		self.play(ShowCreation(vert_line))
		self.play(ApplyMethod(vert_line.shift, RIGHT))

	def func_to_graph(self,x):
		return x


class AddUpdater3(Scene):
	def construct(self):
		dot = Dot()
		text = TextMobject("Label")\
		       .next_to(dot, RIGHT, buff=SMALL_BUFF)
		self.add(dot, text)

		def update_text(obj):
			obj.next_to(dot, RIGHT, buff=SMALL_BUFF)
		
		self.play(
			dot.shift, UP*2,
			UpdateFromFunc(text,update_text)
		)
		self.play(dot.shift, LEFT*2)

		self.wait()

class UpdateValueTracker1(Scene):
	def construct(self):
		theta = ValueTracker(PI/2)
		line_1 = Line(ORIGIN, RIGHT*3,color=RED)
		line_2 = Line(ORIGIN, RIGHT*3,color=GREEN)

		line_2.rotate(theta.get_value(), about_point=ORIGIN)
		line_2.add_updater(lambda m: m.set_angle(theta.get_value()))

		self.add(line_1, line_2)
		self.play(theta.increment_value, PI/2)
		self.wait()

class VectorFieldScene1(Scene):
	def construct(self):
		func = lambda p: np.array([
			p[0]/2,  # x
			p[1]/2,  # y
			0        # z
		])
		# Normalized
		vector_field_norm = VectorField(func)
		# Not normalized
		vector_field_not_norm = VectorField(func, length_func=linear)
		self.play(*[GrowArrow(vec) for vec in vector_field_norm])
		self.wait(2)
		self.play(ReplacementTransform(vector_field_norm,vector_field_not_norm))
		self.wait(2)

def position_tip(self, tip, at_start=False):
	def phi_of_vector(vector):
		xy = complex(*vector[:2])
		if xy == 0:
				return 0;
		a = ((vector[:1])**2 + (vector[1:2])**2)**(1/2)
		vector[0] = a
		vector[1] = vector[2]
		return np.angle(complex(*vector[:2]))
	# Last two control points, defining both
	# the end, and the tangency direction
	if at_start:
		anchor = self.get_start()
		handle = self.get_first_handle()
	else:
		handle = self.get_last_handle()
		anchor = self.get_end()
	tip.rotate(
		angle_of_vector(handle - anchor) -
		PI - tip.get_angle()
	)
	angle = angle_of_vector(handle - anchor) + PI/2
	a = np.array([np.cos(angle),np.sin(angle),0])
	tip.rotate(-phi_of_vector(handle - anchor),a)

	tip.shift(anchor - tip.get_tip_point())
	return tip

class saddlepatch(ThreeDScene):
	def saddle(self, x,y):
		return np.array([x,y,x*y])
	def vector_field_2D(self, func, boundaries):
		field = np.array([func(np.array([x,y]))
			for x in np.arange(boundaries["u_min"], boundaries["u_max"],2)
			for y in np.arange(boundaries["v_min"], boundaries["v_max"],2)
		])
		return field
	def construct(self):

		# quick settings:
		ambientMovement = False
		ambientMovementSpeed = .3
		sexySurfaceAnimation = False

		Φ = 60
		Θ = 180-15

		text_Φ = TextMobject(f"$\\phi={Φ}$").to_corner(UL)
		text_Θ = TextMobject(f"$\\theta={Θ}$").next_to(text_Φ, DOWN)
		self.add_fixed_in_frame_mobjects(text_Φ)
		self.add_fixed_in_frame_mobjects(text_Θ)
		
		self.play(Write(text_Φ), Write(text_Θ))

		TipableVMobject.position_tip = position_tip
		# self.set_camera_orientation(0.6, -PI*0.8, 100)
		# self.move_camera(0.1, -np.pi/2, 10000)
		
		axis = ThreeDAxes()
		axis.add(axis.get_axis_labels())
		self.play(ShowCreation(axis))
		self.move_camera(phi=Φ*DEGREES,theta=Θ*DEGREES,run_time=3)
		if ambientMovement: self.begin_ambient_camera_rotation(rate=ambientMovementSpeed)
		boundaries = {
			"u_min": -PI, "u_max": PI,
			"v_min": -PI, "v_max": PI,
		}
		saddle_surface = ParametricSurface(self.saddle,
			**boundaries, fill_opacity=1
			# checkerboard_colors=[DARK_BLUE, PURPLE_E]
		)
		if sexySurfaceAnimation: self.play(Write(saddle_surface))
		else: self.add(saddle_surface)
		
		curveD=ParametricFunction(
			lambda t : np.array([t,np.sin(t),0]),
			color=YELLOW,t_min=-PI,t_max=PI,
		)
		curveD_perm = curveD.copy()
		curveM=ParametricFunction(
			lambda t : self.saddle(t,np.sin(t)),
			color=RED,t_min=-PI,t_max=PI,
		)
		self.play(ShowCreation(curveD))
		self.add(curveD_perm)
		self.play(Transform(curveD, curveM), run_time=3)

		_t = ValueTracker(-PI)
		t = _t.get_value
		firstpointD = [t(), math.sin(t()), 0]
		firstpointM = self.saddle(firstpointD[0], firstpointD[1])
		d = Dot(firstpointD, radius=DEFAULT_SMALL_DOT_RADIUS, color = GOLD)
		d3 = Dot(firstpointM, radius=DEFAULT_DOT_RADIUS, color = RED)
		
		text_z_eq_xy = TextMobject("$z=x\cdot y$")
		# text_coords = TexMobject("α(t)=(", 
		# 	DecimalNumber(0, num_decimal_places=1, unit=","),
		# 	DecimalNumber(0, num_decimal_places=1, unit=","),
		# 	DecimalNumber(0, num_decimal_places=1),
		# 	")").next_to(text_z_eq_xy, DOWN)
		# text_eq = VGroup(text_z_eq_xy, text_coords)
		# self.add_fixed_in_frame_mobjects(text_eq)
		# text_eq.to_corner(UL)
		# self.play(Write(text_eq))

		self.play(ShowCreation(d), ShowCreation(d3))
		self.wait()
		def updater(m):
			x = t()
			y = sin(x)
			m.move_to(np.array([x,y,0]))
		def updater3d(m):
			x = t()
			y = sin(x)
			saddle_coord = self.saddle(x,y)
			m.move_to(saddle_coord)
			# text_eq.
			# screenpos = np.array([self.coords_to_point(i, saddle_coord) for i in [0,1]])
			# text_z_eq_xy.next_to(d3, RIGHT)
		d.add_updater(updater)
		d3.add_updater(updater3d)
		# self.play(Animation(t.increment_value, 2*PI), ShowCreation(curveM))

		_x, _y, _z = lambda t:t, lambda t:sin(t), lambda t:t*cos(t)
		α = lambda t: np.array([t,sin(t),t*sin(t)])
		αPrime = lambda t: np.array([1, cos(t), sin(t)+t*cos(t)])
		αDoubleprime = lambda t: np.array([0,sin(t), 2*cos(t)-t*sin(t)])
		U = lambda t: np.array([-_y(t), -_x(t), 1])/np.sqrt(_x(t)**2+_y(t)**2+1)
		αDoubleprime_dotU = lambda t: np.dot(αDoubleprime(t), U(t))*U(t)/np.linalg.norm(U(t))
		S_αPrime = lambda t: -np.array([-cos(t), -1, 0]) # (-sin, -t, 1)' = (-cos, -1, 0)
		
		v_αPrime = Vector(αPrime(t())).shift(α(t())).set_color(RED)
		v_αDoubleprime = Vector(αDoubleprime(t())).shift(α(t())).set_color(PURPLE)
		v_U = Vector(U(t())).shift(α(t())).set_color(ORANGE)
		v_αDoubleprime_dotU = DashedVMobject(Vector(αDoubleprime_dotU(t()), tip_length=0).shift(α(t())).set_color(PURPLE))
		v_S_αPrime = Vector(S_αPrime(t())).shift(α(t())).set_color(GREEN)

		text_α = TextMobject("$\\alpha$").to_corner(UR).set_color(RED)
		text_v_αPrime = TextMobject("$\\alpha'$").next_to(text_α, DOWN).set_color(RED)
		text_v_αDoubleprime = TextMobject("$\\alpha''$").next_to(text_v_αPrime, DOWN).set_color(PURPLE)
		text_v_U = TextMobject("$U$").next_to(text_v_αDoubleprime, DOWN).set_color(ORANGE)
		text_S_αPrime = TextMobject("$S(\\alpha')$").next_to(text_v_U, DOWN).set_color(GREEN)

		self.add_fixed_in_frame_mobjects(text_α)
		self.add_fixed_in_frame_mobjects(text_v_αPrime)
		self.add_fixed_in_frame_mobjects(text_v_αDoubleprime)
		self.add_fixed_in_frame_mobjects(text_v_U)
		self.add_fixed_in_frame_mobjects(text_S_αPrime)

		self.play(AnimationGroup(
			ShowCreation(v_αPrime),
			ShowCreation(v_αDoubleprime),
			ShowCreation(v_U),
			ShowCreation(v_αDoubleprime_dotU),
			ShowCreation(v_S_αPrime),
			Write(text_α),
			Write(text_v_αPrime),
			Write(text_v_αDoubleprime),
			Write(text_v_U),
			Write(text_S_αPrime)
		))
		v_αPrime.add_updater(lambda m: m.become(
				Vector(αPrime(t())).shift(α(t())).set_color(m.get_color())
			)
		)
		v_αDoubleprime.add_updater(lambda m: m.become(
				Vector(αDoubleprime(t())).shift(α(t())).set_color(m.get_color())
			)
		)
		v_U.add_updater(lambda m: m.become(
				Vector(U(t())).shift(α(t())).set_color(m.get_color())
			)
		)
		v_αDoubleprime_dotU.add_updater(lambda m: m.become(
				DashedVMobject(Vector(αDoubleprime_dotU(t()),tip_length=0).shift(α(t())).set_color(m.get_color()))
			)
		)
		v_S_αPrime.add_updater(lambda m: m.become(
				Vector(S_αPrime(t())).shift(α(t())).set_color(m.get_color())
			)
		)
		self.play(_t.increment_value, 2*PI, run_time=10)
		self.wait(2)

		# def U(p):
		# 	x,y = p[0],p[1]
		# 	gradient_g = np.array([-y,-x,1])
		# 	return gradient_g/np.linalg.norm(gradient_g)
		# def U_at_p(u,p):
		# 	u = Vector(u)
		# 	UatP = u.shift(np.array([p[0], p[1], p[0]*p[1]]))
		# 	return UatP
		# normal_vector_field = self.vector_field_2D(lambda p: U(p)/2, boundaries)
		# p = self.vector_field_2D(lambda p: p, boundaries)
		# normal_vector_field_atP = [U_at_p(u,p) for u,p in zip(normal_vector_field,p)]
		# # self.play(*[GrowArrow(vec) for vec in normal_vector_field_atP])
		# # self.wait(2)
