# Manim notes

- Easy installation. Use docker install, clone the repo in home folder and add this to .zshrc or .bashrc  
```zsh
manim () {
  export INPUT_PATH=$(realpath .); export OUTPUT_PATH=$(realpath .); docker-compose -f ~/manim/docker-compose.yml run manim $@
}
```

- Use `self.wait(seconds)` to wait a certain amount of time between animations  

- Putting together text and accessing seperate parts:  
```py
second_eq = ["$J(\\theta_{0}, \\theta_{1})$", "=", "$\\frac{1}{2m}$", "$\\sum\\limits_{i=1}^m$", "(", "$h_{\\theta}(x^{(i)})$", "-", "$y^{(i)}$", "$)^2$"]
second_mob = TextMobject(*second_eq)
for i,item in enumerate(second_mob):
	if(i != 0):
		item.next_to(second_mob[i-1],RIGHT)
eq2 = VGroup(*second_mob)

# This will set first item of the equation purple
for i, item in enumerate(eq2):
	if (i<2):
		eq2[i].set_color(color=PURPLE)
	else:
		eq2[i].set_color(color="#00FFFF")
```

Graphscene
```python
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
		self.setup_axes(animate=True)
```

Making a graph
```py
func_graph=self.get_graph(lambda x: x**2,self.function_color)
graph_lab = self.get_graph_label(func_graph, label = "x^{2}")
self.play(ShowCreation(func_graph), Write(graph_lab))
```

Translating from graphscene to view:
```py
	x = self.coords_to_point(1, self.func_to_graph(1))
	y = self.coords_to_point(0, self.func_to_graph(1))
	horz_line = Line(x,y, color=YELLOW)
```


Applying transformations in smooth motion:

```py
def rotate(self, angle, axis=OUT, **kwargs):
        rot_matrix = rotation_matrix(angle, axis)
        self.apply_points_function_about_point(
            lambda points: np.dot(points, rot_matrix.T),
            **kwargs
        )
        return self
```