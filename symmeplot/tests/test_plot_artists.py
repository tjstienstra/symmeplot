import pytest
from symmeplot.plot_artists import Line3D, Vector3D, Circle3D
from symmeplot.tests.utilities import mpl3d_image_comparison
from matplotlib.testing.decorators import cleanup
from matplotlib.pyplot import subplots


class TestPoint3D:
    @pytest.fixture
    def setup_basic(self):
        self.fig, self.ax = subplots(subplot_kw={'projection': '3d'})
        self.p1 = Line3D([.5, .2, .6], color='g')
        self.p2 = Line3D([.2, .4, .9], color='k')
        self.ax.add_artist(self.p1)
        self.ax.add_artist(self.p2)

    @cleanup
    @mpl3d_image_comparison(['point3d_basic.png'])
    def test_basic(self, setup_basic):
        pass

    @cleanup
    @mpl3d_image_comparison(['point3d_update.png'])
    def test_update(self, setup_basic):
        self.p1.update_data([0.9, 0.3, 0.2])


class TestVector3D:
    @pytest.fixture
    def setup_basic(self):
        self.fig, self.ax = subplots(subplot_kw={'projection': '3d'})
        self.v1 = Vector3D([0, 0, 0], [.5, .5, .5], color='g')
        self.v2 = Vector3D([.2, .3, .1], [.7, .6, .8], color='k',
                           mutation_scale=10, arrowstyle='-|>', shrinkA=0,
                           shrinkB=0, picker=20)
        self.ax.add_artist(self.v1)
        self.ax.add_artist(self.v2)

    @cleanup
    @mpl3d_image_comparison(['vector3d_basic.png'])
    def test_basic(self, setup_basic):
        pass

    @cleanup
    @mpl3d_image_comparison(['vector3d_update.png'])
    def test_update(self, setup_basic):
        self.v1.update_data([0.9, 0.1, 0.5], [0, 0.9, -0.1])


class TestCircle3D:
    @pytest.fixture
    def setup_basic(self):
        self.fig, self.ax = subplots(subplot_kw={'projection': '3d'})
        self.c1 = Circle3D([0, 0.5, 0], 0.3, [1, 1, 1], color='g')
        self.c2 = Circle3D([0.5, 0.5, 1], 0.3, [-1, 0.5, 1], color='k')
        self.ax.add_artist(self.c1)
        self.ax.add_artist(self.c2)

    @cleanup
    @mpl3d_image_comparison(['circle3d_basic.png'])
    def test_basic(self, setup_basic):
        pass

    @cleanup
    @mpl3d_image_comparison(['circle3d_update.png'])
    def test_update(self, setup_basic):
        self.c1.update_data([0.5, 0.5, 0], 0.5, 'z')
