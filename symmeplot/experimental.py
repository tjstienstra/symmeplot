from sympy.physics.mechanics import Point, System

from symmeplot.plotter import SymMePlotter


class ExperimentalPlotter(SymMePlotter):
    """Experimental plotter class."""

    @classmethod
    def from_system(cls, ax, system):
        """Create a plotter from a system."""
        if not isinstance(system, System):
            raise TypeError("System must be a System object.")
        plotter = cls(ax, system.frame, system.origin)
        plotter.add_system(system)
        return plotter

    def add_system(self, system):
        """Add a system to the plotter."""

        def get_points(point, points=None):
            if points is None:
                if not isinstance(point, Point):
                    raise TypeError("Point must be a Point object.")
                points = set()
            points.add(point)
            for neighbour in point._pos_dict:
                if neighbour not in points:
                    get_points(neighbour, points)
            return points

        for body in system.bodies:
            if self.get_plot_object(body) is None:
                self.add_body(body)
        for joint in system.joints:
            if self.get_plot_object(joint.parent_point) is None:
                self.add_point(joint.parent_point)
            if self.get_plot_object(joint.child_point) is None:
                self.add_point(joint.child_point)
            self.add_line([joint.parent_point, joint.child_point])
            parent_interframe = self.get_plot_object(joint.parent_interframe)
            if (parent_interframe is not None and
                    parent_interframe.origin != joint.parent_point):
                self.add_frame(joint.parent_interframe, joint.parent_point)
            child_interframe = self.get_plot_object(joint.child_interframe)
            if (child_interframe is not None and
                    child_interframe.origin != joint.child_point):
                self.add_frame(joint.child_interframe, joint.child_point)

        points = get_points(system.origin)
        for point in points:
            if self.get_plot_object(point) is None:
                self.add_point(point)
