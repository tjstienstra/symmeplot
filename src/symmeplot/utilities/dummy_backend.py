from __future__ import annotations

from typing import TYPE_CHECKING

from symmeplot.core import (
    ArtistBase,
    PlotBase,
    PlotBodyMixin,
    PlotFrameMixin,
    PlotLineMixin,
    PlotPointMixin,
    PlotTracedPointMixin,
    PlotVectorMixin,
    SceneBase,
)

if TYPE_CHECKING:
    from sympy.physics.vector import Point, ReferenceFrame, Vector


class DummyArtist(ArtistBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.update_args = None
        self.visible = True

    def update_data(self, *args: object) -> None:
        self.update_args = args


class _PlotBase(PlotBase):
    def plot(self) -> None:
        self.update()
        for child in self._children:
            child.plot()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, is_visible: bool) -> None:
        for artist, _ in self._artists:
            artist.visible = bool(is_visible)
        for child in self._children:
            child.visible = bool(is_visible)
        self._visible = bool(is_visible)


class PlotPoint(PlotPointMixin, _PlotBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.add_artist(DummyArtist(), self.get_sympy_object_exprs())


class PlotTracedPoint(PlotTracedPointMixin, _PlotBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.add_artist(DummyArtist(), self.get_sympy_object_exprs())

    def update(self) -> None:
        self._update_trace_history()
        for child in self._children:
            child.update()


class PlotLine(PlotLineMixin, _PlotBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.add_artist(DummyArtist(), self.get_sympy_object_exprs())


class PlotVector(PlotVectorMixin, _PlotBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.add_artist(DummyArtist(), self.get_sympy_object_exprs())


class PlotFrame(PlotFrameMixin, _PlotBase):
    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        frame: ReferenceFrame,
        origin: Point | Vector | None = None,
        name: str | None = None,
        scale: float = 0.1,
    ) -> None:
        super().__init__(inertial_frame, zero_point, frame, origin, name, scale)
        for v in self.frame:
            self._children.append(
                PlotVector(self.inertial_frame, self.zero_point, scale * v, self.origin)
            )


class PlotBody(PlotBodyMixin, _PlotBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # Particle.masscenter does not yet exist in SymPy 1.12
        mc = getattr(self.body, "masscenter", getattr(self.body, "point", None))
        self._children.append(PlotPoint(self.inertial_frame, self.zero_point, mc))
        if hasattr(self.body, "frame"):
            self._children.append(
                PlotFrame(
                    self.inertial_frame, self.zero_point, self.body.frame, origin=mc
                )
            )


class Scene3D(SceneBase):
    _PlotPoint: type[PlotBase] = PlotPoint
    _PlotTracedPoint: type[PlotBase] = PlotTracedPoint
    _PlotLine: type[PlotBase] = PlotLine
    _PlotVector: type[PlotBase] = PlotVector
    _PlotFrame: type[PlotBase] = PlotFrame
    _PlotBody: type[PlotBase] = PlotBody
